"""
Temporal LSTM Model for Mental Health Digital Twin.

Architecture:
- Bidirectional LSTM layers with sequence modeling
- Dropout for regularization
- Ensemble with XGBoost for robustness
- PyTorch backend (not Keras)

Input: Sequences of shape (batch_size, lookback_days=14, n_features)
Output: Anxiety severity class probabilities (4 classes)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
import joblib
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import json

import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LSTMModel(nn.Module):
    """
    LSTM-based temporal model for mental health prediction.
    
    Architecture:
    - Input: (seq_len, n_features)
    - LSTM(64, return_sequences=True) + Dropout(0.2)
    - LSTM(32) + Dropout(0.2)
    - Dense(16, relu)
    - Dense(4, softmax) [4 severity classes]
    """
    
    def __init__(self, n_features: int, hidden_size_1: int = 64, 
                 hidden_size_2: int = 32, dropout: float = 0.2, 
                 n_classes: int = 4, bidirectional: bool = True):
        """
        Args:
            n_features: Number of input features
            hidden_size_1: Hidden size of first LSTM layer
            hidden_size_2: Hidden size of second LSTM layer
            dropout: Dropout rate
            n_classes: Number of output classes
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.n_features = n_features
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dropout_rate = dropout
        self.n_classes = n_classes
        self.bidirectional = bidirectional
        
        # First LSTM layer with return_sequences=True
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size_1,
            batch_first=True,
            dropout=dropout if hidden_size_1 > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Second LSTM layer
        lstm1_output_size = hidden_size_1 * (2 if bidirectional else 1)
        self.lstm2 = nn.LSTM(
            input_size=lstm1_output_size,
            hidden_size=hidden_size_2,
            batch_first=True,
            dropout=dropout if hidden_size_2 > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Dense layers
        lstm2_output_size = hidden_size_2 * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm2_output_size, 16)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, n_classes)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, n_features)
        
        Returns:
            Tensor of shape (batch_size, n_classes) with logits
        """
        # First LSTM
        lstm1_out, (h1, c1) = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second LSTM
        lstm2_out, (h2, c2) = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Use last output of LSTM
        lstm_out = lstm2_out[:, -1, :]
        
        # Dense layers
        fc1_out = self.fc1(lstm_out)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout3(fc1_out)
        logits = self.fc2(fc1_out)
        
        return logits


class TemporalDataset:
    """
    Prepares sequential data for LSTM training.
    """
    
    def __init__(self, df: pd.DataFrame, lookback_days: int = 14, 
                 group_col: str = 'user_id', target_col: str = 'Anxiety_Severity'):
        """
        Args:
            df: DataFrame with time series data (must be sorted by date per user)
            lookback_days: Number of days to look back
            group_col: Column for grouping (user_id)
            target_col: Target column name
        """
        self.df = df.sort_values(group_col)
        self.lookback_days = lookback_days
        self.group_col = group_col
        self.target_col = target_col
        self.sequences = []
        self.labels = []
        self.users = []
        
    def build_sequences(self, feature_cols: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sequences from data.
        
        Args:
            feature_cols: List of feature columns to use
        
        Returns:
            Tuple of (sequences, labels) as numpy arrays
        """
        if feature_cols is None:
            feature_cols = config.FINAL_MODEL_FEATURES
        
        for user_id in self.df[self.group_col].unique():
            user_df = self.df[self.df[self.group_col] == user_id].copy()
            user_df = user_df.sort_values('Date') if 'Date' in user_df.columns else user_df
            
            if len(user_df) < self.lookback_days:
                logging.warning(f"User {user_id} has {len(user_df)} records, less than lookback_days={self.lookback_days}")
                continue
            
            X = user_df[feature_cols].values
            y = user_df[self.target_col].values
            
            # Create sliding windows
            for i in range(len(X) - self.lookback_days + 1):
                seq = X[i:i + self.lookback_days]
                label = y[i + self.lookback_days - 1]  # Label at end of sequence
                
                self.sequences.append(seq)
                self.labels.append(label)
                self.users.append(user_id)
        
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
        self.users = np.array(self.users)
        
        logging.info(f"Built {len(self.sequences)} sequences from {len(self.df)} records")
        
        return self.sequences, self.labels
    
    def get_dataloader(self, labels_encoded: np.ndarray, batch_size: int = 32, 
                      shuffle: bool = True, device: str = 'cpu') -> DataLoader:
        """
        Create PyTorch DataLoader.
        
        Args:
            labels_encoded: Encoded labels (0, 1, 2, 3)
            batch_size: Batch size
            shuffle: Whether to shuffle data
            device: Device to load tensors to
        
        Returns:
            PyTorch DataLoader
        """
        X_tensor = torch.FloatTensor(self.sequences).to(device)
        y_tensor = torch.LongTensor(labels_encoded).to(device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader


class LSTMTrainer:
    """
    Handles training and evaluation of LSTM model.
    """
    
    def __init__(self, model: LSTMModel, device: str = 'cpu', 
                 learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """
        Args:
            model: LSTM model instance
            device: 'cpu' or 'cuda'
            learning_rate: Adam learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in dataloader:
            self.optimizer.zero_grad()
            
            # Forward
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, pred = torch.max(logits, 1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Validate on validation set.
        
        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item()
                
                _, pred = torch.max(logits, 1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, 
           epochs: int = 100, patience: int = 10, verbose: bool = True) -> Dict[str, list]:
        """
        Train model with early stopping.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Maximum epochs
            patience: Early stopping patience
            verbose: Print progress
        
        Returns:
            History dict
        """
        logging.info(f"Starting training for {epochs} epochs (early stopping patience={patience})...")
        
        self.patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}/{epochs} | "
                            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.best_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_state)
                    break
        
        return self.history
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Input sequences of shape (n_samples, seq_len, n_features)
        
        Returns:
            Probabilities of shape (n_samples, n_classes)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()


class ModelEnsemble:
    """
    Ensemble combining XGBoost and LSTM predictions.
    """
    
    def __init__(self, xgb_model_path: str = 'mental_twin_xgb_model.pkl',
                 lstm_model_path: str = 'mental_twin_lstm_model.pt',
                 xgb_weight: float = 0.5, lstm_weight: float = 0.5):
        """
        Args:
            xgb_model_path: Path to saved XGBoost model
            lstm_model_path: Path to saved LSTM model
            xgb_weight: Weight for XGBoost predictions
            lstm_weight: Weight for LSTM predictions
        """
        self.xgb_model = None
        self.lstm_model = None
        self.lstm_trainer = None
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Try to load models
        if Path(xgb_model_path).exists():
            try:
                self.xgb_model = joblib.load(xgb_model_path)
                logging.info(f"Loaded XGBoost model from {xgb_model_path}")
            except Exception as e:
                logging.warning(f"Failed to load XGBoost model: {e}")
        
        if Path(lstm_model_path).exists():
            try:
                checkpoint = torch.load(lstm_model_path, map_location=self.device)
                # Assume checkpoint contains model state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Recreate model (need to infer architecture from state_dict or use config)
                n_features = len(config.FINAL_MODEL_FEATURES)
                self.lstm_model = LSTMModel(n_features=n_features).to(self.device)
                self.lstm_model.load_state_dict(state_dict)
                self.lstm_model.eval()
                logging.info(f"Loaded LSTM model from {lstm_model_path}")
            except Exception as e:
                logging.warning(f"Failed to load LSTM model: {e}")
    
    def predict_proba(self, X_tabular: np.ndarray, X_sequences: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Ensemble prediction combining XGBoost and LSTM.
        
        Args:
            X_tabular: Tabular features for XGBoost (n_samples, n_features)
            X_sequences: Sequential features for LSTM (n_samples, seq_len, n_features)
        
        Returns:
            Ensemble probabilities (n_samples, n_classes)
        """
        probs = None
        
        if self.xgb_model is not None:
            try:
                xgb_probs = self.xgb_model.predict_proba(X_tabular)
                if probs is None:
                    probs = self.xgb_weight * xgb_probs
                else:
                    probs = probs + self.xgb_weight * xgb_probs
            except Exception as e:
                logging.warning(f"XGBoost prediction failed: {e}")
        
        if self.lstm_model is not None and X_sequences is not None:
            try:
                lstm_probs = self.lstm_trainer.predict_proba(X_sequences) if self.lstm_trainer else \
                            self._predict_lstm_direct(X_sequences)
                if probs is None:
                    probs = self.lstm_weight * lstm_probs
                else:
                    probs = probs + self.lstm_weight * lstm_probs
            except Exception as e:
                logging.warning(f"LSTM prediction failed: {e}")
        
        # Normalize if both models contributed
        if probs is not None and (self.xgb_model is not None and self.lstm_model is not None):
            probs = probs / (self.xgb_weight + self.lstm_weight)
        
        return probs
    
    def _predict_lstm_direct(self, X: np.ndarray) -> np.ndarray:
        """Direct LSTM prediction without trainer."""
        self.lstm_model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.lstm_model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()


def save_lstm_model(trainer: LSTMTrainer, model_path: str = 'mental_twin_lstm_model.pt',
                   config_path: str = 'mental_twin_lstm_config.json'):
    """
    Save LSTM model and configuration.
    
    Args:
        trainer: Trained LSTMTrainer instance
        model_path: Path to save model
        config_path: Path to save config
    """
    checkpoint = {
        'model_state_dict': trainer.model.state_dict(),
        'history': trainer.history,
    }
    torch.save(checkpoint, model_path)
    logging.info(f"Saved LSTM model to {model_path}")
    
    config_dict = {
        'n_features': trainer.model.n_features,
        'hidden_size_1': trainer.model.hidden_size_1,
        'hidden_size_2': trainer.model.hidden_size_2,
        'dropout': trainer.model.dropout_rate,
        'n_classes': trainer.model.n_classes,
        'bidirectional': trainer.model.bidirectional,
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logging.info(f"Saved LSTM config to {config_path}")


if __name__ == '__main__':
    # Example usage (requires proper data setup)
    logging.info("Temporal model module ready for import")
