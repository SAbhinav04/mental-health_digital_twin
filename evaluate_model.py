"""
Rigorous Evaluation Pipeline for Mental Health Digital Twin.

Implements:
- Leave-One-Subject-Out (LOSO) cross-validation (healthcare ML best practice)
- Stratified 5-fold cross-validation
- Comprehensive metrics: Accuracy, Macro F1, Weighted F1, AUC-ROC, Confusion Matrix, Cohen's Kappa
- Model comparison: XGBoost vs LSTM vs Ensemble
- JSON results export
"""

import pandas as pd
import numpy as np
import json
import logging
import os
import joblib
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    cohen_kappa_score, classification_report, roc_curve, auc
)
import xgboost as xgb
from sklearn.preprocessing import label_binarize

import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RigorousModelEvaluator:
    """
    Comprehensive evaluation framework for multiclass mental health prediction.
    """
    
    def __init__(self, model_names: Optional[List[str]] = None):
        """
        Args:
            model_names: List of model names to compare. Defaults to ['xgboost', 'lstm', 'ensemble'].
        """
        self.model_names = model_names or ['xgboost', 'lstm', 'ensemble']
        self.results = {}
        self.le = LabelEncoder()
        
    def evaluate_leave_one_subject_out(self, 
                                      X: np.ndarray, 
                                      y: np.ndarray, 
                                      groups: np.ndarray,
                                      train_fn,
                                      predict_fn,
                                      model_name: str = 'xgboost') -> Dict[str, Any]:
        """
        Leave-One-Subject-Out (LOSO) cross-validation.
        Critical for healthcare: test on entirely unseen subject.
        
        Args:
            X: Feature matrix
            y: Labels (either encoded or categorical)
            groups: Subject/user IDs (array of same length as X)
            train_fn: Function that trains model: train_fn(X_train, y_train) -> model
            predict_fn: Function that predicts: predict_fn(model, X_test) -> y_pred_proba
            model_name: Name for results tracking
        
        Returns:
            Dictionary with LOSO metrics
        """
        logging.info(f"Starting LOSO cross-validation ({model_name})...")
        
        # Encode labels if needed
        if isinstance(y[0], str):
            y_encoded = self.le.fit_transform(y.astype(str))
        else:
            y_encoded = y
            self.le.fit(np.unique(y_encoded))
        
        loso = LeaveOneGroupOut()
        fold_results = []
        y_true_all = []
        y_pred_all = []
        y_pred_proba_all = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(loso.split(X, y_encoded, groups)):
            logging.info(f"  LOSO fold {fold_idx + 1}/{len(np.unique(groups))}: "
                        f"train_subjects={len(np.unique(groups[train_idx]))}, "
                        f"test_subject={groups[test_idx[0]]}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            
            # Train model
            model = train_fn(X_train, y_train)
            
            # Predict
            y_pred_proba = predict_fn(model, X_test)
            y_pred = np.argmax(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else y_pred_proba
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_pred_proba_all.extend(y_pred_proba if y_pred_proba.ndim > 1 else np.eye(len(self.le.classes_))[y_pred])
            
            # Fold metrics
            fold_acc = accuracy_score(y_test, y_pred)
            fold_results.append({'fold': fold_idx, 'accuracy': fold_acc})
        
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        y_pred_proba_all = np.array(y_pred_proba_all)
        
        # Aggregate metrics
        metrics = self._compute_metrics(y_true_all, y_pred_all, y_pred_proba_all)
        metrics['cv_type'] = 'LOSO'
        metrics['num_folds'] = len(np.unique(groups))
        metrics['folds'] = fold_results
        
        return metrics
    
    def evaluate_stratified_kfold(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 k: int = 5,
                                 train_fn = None,
                                 predict_fn = None,
                                 model_name: str = 'xgboost') -> Dict[str, Any]:
        """
        Stratified k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            k: Number of folds
            train_fn: Training function
            predict_fn: Prediction function
            model_name: Model name for logging
        
        Returns:
            Dictionary with stratified k-fold metrics
        """
        logging.info(f"Starting {k}-fold stratified cross-validation ({model_name})...")
        
        # Encode labels
        if isinstance(y[0], str):
            y_encoded = self.le.fit_transform(y.astype(str))
        else:
            y_encoded = y
            self.le.fit(np.unique(y_encoded))
        
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_results = []
        y_true_all = []
        y_pred_all = []
        y_pred_proba_all = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
            logging.info(f"  Fold {fold_idx + 1}/{k}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            
            # Train
            model = train_fn(X_train, y_train)
            
            # Predict
            y_pred_proba = predict_fn(model, X_test)
            y_pred = np.argmax(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else y_pred_proba
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_pred_proba_all.extend(y_pred_proba if y_pred_proba.ndim > 1 else np.eye(len(self.le.classes_))[y_pred])
            
            fold_acc = accuracy_score(y_test, y_pred)
            fold_results.append({'fold': fold_idx, 'accuracy': fold_acc})
        
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        y_pred_proba_all = np.array(y_pred_proba_all)
        
        # Aggregate metrics
        metrics = self._compute_metrics(y_true_all, y_pred_all, y_pred_proba_all)
        metrics['cv_type'] = f'{k}-Fold Stratified'
        metrics['num_folds'] = k
        metrics['folds'] = fold_results
        
        return metrics
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels (encoded)
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (n_samples, n_classes)
        
        Returns:
            Dictionary of metrics
        """
        num_classes = len(self.le.classes_)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        
        # AUC-ROC (one-vs-rest for multiclass)
        auc_roc = None
        if y_pred_proba is not None and num_classes > 2:
            try:
                y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
                auc_roc = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                logging.warning(f"Could not compute AUC-ROC: {e}")
        elif y_pred_proba is not None and num_classes == 2:
            try:
                auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1])
            except Exception:
                pass
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, 
                                            labels=np.arange(num_classes),
                                            target_names=self.le.classes_,
                                            output_dict=True,
                                            zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'cohens_kappa': float(kappa),
            'auc_roc': float(auc_roc) if auc_roc is not None else None,
            'confusion_matrix': cm.tolist(),
            'class_report': class_report,
        }
        
        return metrics
    
    def compare_models(self, 
                      results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create comparison table across models.
        
        Args:
            results_dict: Dict mapping model names to their evaluation results
        
        Returns:
            Pandas DataFrame with comparison
        """
        comparison_data = []
        
        for model_name, metrics in results_dict.items():
            row = {
                'Model': model_name,
                'CV Method': metrics.get('cv_type', 'N/A'),
                'Accuracy': metrics.get('accuracy', 0),
                'Macro F1': metrics.get('macro_f1', 0),
                'Weighted F1': metrics.get('weighted_f1', 0),
                "Cohen's Kappa": metrics.get('cohens_kappa', 0),
                'AUC-ROC': metrics.get('auc_roc', 'N/A'),
            }
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        return df_comparison
    
    def print_formatted_summary(self, 
                               results_dict: Dict[str, Dict[str, Any]],
                               output_file: Optional[str] = None):
        """
        Print formatted summary table and save to file.
        
        Args:
            results_dict: Model results dictionary
            output_file: Optional file to save summary
        """
        df_comparison = self.compare_models(results_dict)
        
        print("\n" + "="*100)
        print("MODEL EVALUATION SUMMARY")
        print("="*100)
        print(df_comparison.to_string(index=False))
        print("="*100)
        
        # Per-model details
        for model_name, metrics in results_dict.items():
            print(f"\n--- {model_name.upper()} ---")
            print(f"CV Method: {metrics.get('cv_type', 'N/A')} ({metrics.get('num_folds', 'N/A')} folds)")
            print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"Macro F1: {metrics.get('macro_f1', 0):.4f}")
            print(f"Weighted F1: {metrics.get('weighted_f1', 0):.4f}")
            print(f"Cohen's Kappa: {metrics.get('cohens_kappa', 0):.4f}")
            if metrics.get('auc_roc') is not None:
                print(f"AUC-ROC: {metrics.get('auc_roc'):.4f}")
            
            # Confusion matrix
            cm = np.array(metrics.get('confusion_matrix', []))
            if cm.size > 0:
                print(f"\nConfusion Matrix:")
                print(cm)
            
            # Per-class metrics
            class_report = metrics.get('class_report', {})
            if class_report:
                print(f"\nPer-Class Metrics:")
                for class_name in self.le.classes_:
                    if class_name in class_report:
                        cr = class_report[class_name]
                        print(f"  {class_name}: Precision={cr.get('precision', 0):.4f}, "
                              f"Recall={cr.get('recall', 0):.4f}, F1={cr.get('f1-score', 0):.4f}")
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write("MODEL EVALUATION SUMMARY\n")
                f.write("="*100 + "\n")
                f.write(df_comparison.to_string(index=False))
                f.write("\n" + "="*100 + "\n")
            logging.info(f"Summary saved to {output_file}")


def evaluate_xgboost(df: pd.DataFrame, 
                     target_col: str = 'Anxiety_Severity',
                     group_col: str = 'user_id',
                     test_size: float = 0.1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate XGBoost model with both LOSO and stratified 5-fold CV.
    
    Args:
        df: Training dataframe
        target_col: Target column name
        group_col: Column for grouping (subject ID)
        test_size: Test set size (for baseline)
    
    Returns:
        Tuple of (loso_results, stratified_cv_results)
    """
    logging.info("Evaluating XGBoost model...")
    
    # Prepare data
    feature_cols = config.FINAL_MODEL_FEATURES
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df[group_col].values
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    
    # Define train/predict functions for XGBoost
    def train_xgb(X_train, y_train):
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(np.unique(y_train)),
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train)
        return xgb_model
    
    def predict_xgb(model, X_test):
        # Get probability predictions
        return model.predict_proba(X_test)
    
    # Evaluator
    evaluator = RigorousModelEvaluator()
    evaluator.le = le
    
    # LOSO
    loso_results = evaluator.evaluate_leave_one_subject_out(
        X, y_encoded, groups, train_xgb, predict_xgb, model_name='xgboost'
    )
    
    # Stratified 5-fold
    stratified_results = evaluator.evaluate_stratified_kfold(
        X, y_encoded, k=5, train_fn=train_xgb, predict_fn=predict_xgb, model_name='xgboost'
    )
    
    return loso_results, stratified_results, evaluator


if __name__ == '__main__':
    # Example usage
    from dataset_loader import DatasetManager, get_feature_aligned_dataframe
    
    manager = DatasetManager(use_real_data=False)  # Use synthesized for testing
    df, metadata = manager.load_data()
    
    print(f"Loaded {len(df)} records from {metadata['source']}")
    
    # Align features
    df = get_feature_aligned_dataframe(df, config.FINAL_MODEL_FEATURES)
    
    # Evaluate XGBoost
    loso_res, strat_res, evaluator = evaluate_xgboost(df)
    
    results = {
        'xgboost_loso': loso_res,
        'xgboost_stratified': strat_res,
    }
    
    evaluator.print_formatted_summary(results, output_file='evaluation_summary.txt')
    
    # Save to JSON
    def serialize_results(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=serialize_results)
    
    logging.info("Evaluation complete. Results saved to evaluation_results.json")
