"""
Dataset Loader for Mental Health Digital Twin.

Supports loading:
1. Real Samsung HRV + Sleep Diary dataset (Nature Scientific Data 2025)
2. Synthesized data as fallback

Features computed:
- HRV metrics (SDNN, SDSD, LF, HF, RMSSD)
- Sleep metrics (duration, consistency, debt)
- Activity metrics (steps, trends)
- GAD-7 severity labels
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from datetime import datetime, timedelta
import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SamsungHRVDatasetLoader:
    """
    Loads and processes Samsung HRV + Sleep Diary dataset.
    
    Expected directory structure:
    samsung_dataset/
    ├── participant_001/
    │   ├── hrv_metrics.csv        (Date, SDNN, SDSD, LF, HF, RMSSD, RHRV, LFHF)
    │   ├── sleep_diary.csv        (Date, sleep_onset_time, sleep_duration, awakenings)
    │   ├── questionnaire.csv      (Date, GAD7_score, PHQ9_score, ISI_score)
    │   └── activity.csv           (Date, steps, distance)
    ├── participant_002/
    ...
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Args:
            dataset_path: Path to Samsung dataset root. If None, uses SAMSUNG_DATASET_PATH from config.
        """
        self.dataset_path = dataset_path or getattr(config, 'SAMSUNG_DATASET_PATH', None)
        self.participants = []
        
    def load_dataset(self) -> pd.DataFrame:
        """
        Load and preprocess real Samsung dataset.
        
        Returns:
            DataFrame with aligned features and GAD-7 severity labels
        """
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            logging.warning(f"Samsung dataset path not found: {self.dataset_path}")
            return None
            
        logging.info(f"Loading Samsung HRV dataset from {self.dataset_path}")
        
        all_data = []
        participant_dirs = sorted([d for d in os.listdir(self.dataset_path) 
                                   if os.path.isdir(os.path.join(self.dataset_path, d))])
        
        for participant_id in participant_dirs:
            try:
                participant_data = self._load_participant(participant_id)
                if participant_data is not None:
                    all_data.append(participant_data)
                    logging.info(f"Loaded participant {participant_id}: {len(participant_data)} records")
            except Exception as e:
                logging.warning(f"Failed to load participant {participant_id}: {e}")
                
        if not all_data:
            logging.warning("No participant data loaded from Samsung dataset")
            return None
            
        df = pd.concat(all_data, ignore_index=True)
        logging.info(f"Combined dataset shape: {df.shape}")
        return df
    
    def _load_participant(self, participant_id: str) -> Optional[pd.DataFrame]:
        """Load data for a single participant."""
        participant_dir = os.path.join(self.dataset_path, participant_id)
        
        # Load individual CSV files
        hrv_df = self._safe_load_csv(os.path.join(participant_dir, 'hrv_metrics.csv'))
        sleep_df = self._safe_load_csv(os.path.join(participant_dir, 'sleep_diary.csv'))
        questionnaire_df = self._safe_load_csv(os.path.join(participant_dir, 'questionnaire.csv'))
        activity_df = self._safe_load_csv(os.path.join(participant_dir, 'activity.csv'))
        
        if hrv_df is None:
            return None
            
        # Ensure date columns are datetime
        for df in [hrv_df, sleep_df, questionnaire_df, activity_df]:
            if df is not None and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
        
        # Align all data by date
        merged_df = hrv_df.copy()
        
        if sleep_df is not None:
            merged_df = merged_df.merge(sleep_df, on='Date', how='left')
        if activity_df is not None:
            merged_df = merged_df.merge(activity_df, on='Date', how='left')
        
        # Align questionnaire (typically biweekly, so forward-fill to next questionnaire date)
        if questionnaire_df is not None:
            merged_df = merged_df.merge(questionnaire_df, on='Date', how='left')
            merged_df['GAD7_score'] = merged_df['GAD7_score'].fillna(method='ffill')
            merged_df['GAD7_score'] = merged_df['GAD7_score'].fillna(method='bfill')
        else:
            logging.warning(f"No questionnaire data for {participant_id}")
            return None
            
        merged_df['user_id'] = participant_id
        
        # Drop rows with missing GAD-7 scores
        merged_df = merged_df.dropna(subset=['GAD7_score'])
        
        # Compute aggregated features
        merged_df = self._compute_aggregate_features(merged_df)
        
        # Map GAD-7 to severity labels
        merged_df['Anxiety_Severity'] = merged_df['GAD7_score'].apply(self._gad7_to_severity)
        
        return merged_df
    
    def _safe_load_csv(self, filepath: str) -> Optional[pd.DataFrame]:
        """Safely load CSV file, return None if not found."""
        if not os.path.exists(filepath):
            return None
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            logging.warning(f"Failed to load {filepath}: {e}")
            return None
    
    def _compute_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling 7-day and 14-day aggregate features per user."""
        df = df.sort_values('Date')
        
        # HRV metrics (ensure column names match expected)
        hrv_cols = ['SDNN', 'SDSD', 'RMSSD', 'LF', 'HF']
        existing_hrv = [col for col in hrv_cols if col in df.columns]
        
        if existing_hrv:
            for col in existing_hrv:
                df[f'{col}_7d_mean'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_7d_std'] = df[col].rolling(window=7, min_periods=1).std().fillna(0)
                df[f'{col}_14d_mean'] = df[col].rolling(window=14, min_periods=1).mean()
                df[f'{col}_14d_trend'] = df[col].rolling(window=14, min_periods=1).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False)
        
        # Sleep metrics
        if 'sleep_duration' in df.columns:
            df['sleep_duration_7d_mean'] = df['sleep_duration'].rolling(window=7, min_periods=1).mean()
            df['sleep_consistency_7d'] = df['sleep_duration'].rolling(window=7, min_periods=1).std().fillna(0)
        
        if 'sleep_onset_time' in df.columns:
            # Compute std of sleep onset times (consistency)
            df['sleep_onset_consistency_14d'] = df['sleep_onset_time'].rolling(window=14, min_periods=1).std().fillna(0)
        
        # Sleep debt
        if 'sleep_duration' in df.columns:
            sleep_7d_avg = df['sleep_duration'].rolling(window=7, min_periods=1).mean()
            df['sleep_debt_14d'] = (sleep_7d_avg - df['sleep_duration']).rolling(window=14, min_periods=1).sum().fillna(0)
        
        # Activity metrics
        if 'steps' in df.columns:
            df['steps_7d_mean'] = df['steps'].rolling(window=7, min_periods=1).mean()
            df['activity_consistency_7d'] = df['steps'].rolling(window=7, min_periods=1).std().fillna(0) / (df['steps_7d_mean'].fillna(1) + 1e-8)  # Coefficient of variation
        
        return df
    
    def _gad7_to_severity(self, score: float) -> str:
        """Map GAD-7 score to severity label."""
        if pd.isna(score):
            return 'Unknown'
        if score < 5:
            return 'Minimal'
        elif score < 10:
            return 'Mild'
        elif score < 15:
            return 'Moderate'
        else:
            return 'Severe'


class DatasetManager:
    """
    Unified interface to load either real or synthesized data.
    Falls back to synthesized data if real dataset is not available.
    """
    
    def __init__(self, use_real_data: bool = True):
        """
        Args:
            use_real_data: If True, try to load real dataset; fallback to synthesized.
        """
        self.use_real_data = use_real_data
        
    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Load training data.
        
        Returns:
            Tuple of (DataFrame, metadata dict with source info)
        """
        metadata = {}
        
        if self.use_real_data:
            df = self._load_real_data()
            if df is not None:
                metadata['source'] = 'Samsung HRV + Sleep Diary Dataset'
                metadata['note'] = 'Real clinical data (Nature Scientific Data 2025)'
                return df, metadata
            else:
                logging.info("Real dataset not available, falling back to synthesized data")
        
        df = self._load_synthesized_data()
        metadata['source'] = 'Synthesized Data'
        metadata['note'] = 'Synthetic data for development/testing'
        return df, metadata
    
    def _load_real_data(self) -> Optional[pd.DataFrame]:
        """Load real Samsung dataset."""
        loader = SamsungHRVDatasetLoader()
        return loader.load_dataset()
    
    def _load_synthesized_data(self) -> pd.DataFrame:
        """Load synthesized data using existing pipeline."""
        try:
            from data_simulation_and_preprocessing import load_and_combine_data, inject_extreme_periods, \
                align_mood_sentiment, feature_engineering
            
            logging.info("Generating synthesized data...")
            df = load_and_combine_data('raw_data', {})
            df = inject_extreme_periods(df)
            df = align_mood_sentiment(df)
            df = feature_engineering(df)
            
            return df
        except Exception as e:
            logging.error(f"Failed to load synthesized data: {e}")
            return None


def get_feature_aligned_dataframe(df: pd.DataFrame, target_features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Ensure DataFrame matches expected feature schema.
    Fill missing features with column means.
    
    Args:
        df: Input DataFrame
        target_features: Expected feature list (defaults to config.FINAL_MODEL_FEATURES)
    
    Returns:
        DataFrame with exactly the target features
    """
    if target_features is None:
        target_features = config.FINAL_MODEL_FEATURES
    
    # Fill missing features with 0 or column mean
    for feature in target_features:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Select only target features
    df = df[target_features].copy()
    
    # Fill any remaining NaN with column means
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    
    return df


if __name__ == '__main__':
    # Example usage
    manager = DatasetManager(use_real_data=True)
    df, metadata = manager.load_data()
    
    print(f"\nDataset Source: {metadata['source']}")
    print(f"Note: {metadata['note']}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
