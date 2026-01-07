import pandas as pd
import numpy as np
import os
import logging
import datetime 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


RAW_DATA_DIR = 'raw_data'
OUTPUT_FILE = 'processed_mental_twin_data.csv'
EPSILON = 1e-8 
analyzer = SentimentIntensityAnalyzer()
MOOD_LOGS = {
    'POSITIVE': ["Feeling fantastic and productive!", "Everything is calm and joyful.", "Had an amazing, relaxing day.", "Very happy with my progress.", "Life is good."],
    'NEGATIVE': ["Extremely anxious about a deadline.", "I'm exhausted and stressed out.", "Everything is falling apart.", "Feeling overwhelmed and drained.", "Awful mood, feeling sad."],
    'NEUTRAL': ["An okay day, nothing special.", "Routine day, uneventful.", "A bit tired today.", "Got things done, nothing more.", "Standard day."],
}

def get_sentiment_score(text):
    if not text: return 0.0
    return analyzer.polarity_scores(text)['compound']

def generate_user_data(user_id, profile):
    """Generates a full history dataframe for a single user profile."""
    logging.info(f"Generating data for User: {user_id} ({profile['description']})")
    num_days = 200 
    date_range = pd.date_range(start='2024-01-01', periods=num_days, freq='D')
    
    df = pd.DataFrame({'Date': date_range.date})
    df['Date'] = pd.to_datetime(df['Date'])
    df['user_id'] = user_id

    np.random.seed(hash(user_id) % 1000) 


    df['hrv_mean'] = np.random.normal(profile['hrv_mu'], profile['hrv_sigma'], size=num_days)
    df['bp_systolic_mean'] = np.random.normal(profile['bp_sys_mu'], profile['bp_sys_sigma'], size=num_days)
    df['bp_diastolic_mean'] = np.random.normal(profile['bp_sys_mu'] * 0.67, profile['bp_sys_sigma'] * 0.5, size=num_days)
    df['sleep_duration_sum'] = np.random.normal(profile['sleep_mu'], profile['sleep_sigma'], size=num_days)
    df['steps_sum'] = np.random.normal(profile['steps_mu'], profile['steps_sigma'], size=num_days)
    df['audio_level_mean'] = np.random.normal(70, 5, size=num_days)
    

    df['distance_sum'] = df['steps_sum'] * 0.7 
    df['energy_active_sum'] = df['steps_sum'] * 0.05 
    df['flights_climbed_sum'] = df['steps_sum'] * 0.01 
    df['energy_basal_mean'] = np.random.normal(1800, 50, size=num_days)
    df['audio_level_std'] = np.random.normal(5, 2, size=num_days)
    df['walking_symmetry_mean'] = np.random.normal(50, 1, size=num_days)
    
    return df

def load_and_combine_data(data_dir, mapping):
    """Generates and combines ALL User Data for training purposes."""
    all_user_data = []
    
    for user_id, profile in config.USER_PROFILES.items():
        user_df = generate_user_data(user_id, profile)
        all_user_data.append(user_df)

    final_combined_df = pd.concat(all_user_data, ignore_index=True)
    
    return final_combined_df

def inject_extreme_periods(df):
    """
    Artificially introduces periods of extreme health and extreme stress, 
    performed PER USER.
    """
    logging.info("Injecting periods of extreme health and stress...")
    
    final_injected_df = []
    
    for user_id in df['user_id'].unique():
        user_df = df[df['user_id'] == user_id].copy()
        
        if len(user_df) < 20:
             final_injected_df.append(user_df)
             continue
             
        extreme_df = user_df.copy()
        unique_dates = extreme_df['Date'].unique()

        
        stress_dates = np.random.choice(unique_dates, size=5, replace=False)
        for date in stress_dates:
            idx = extreme_df[extreme_df['Date'] == date].index
            extreme_df.loc[idx, 'bp_systolic_mean'] += 15
            extreme_df.loc[idx, 'hrv_mean'] -= 20         
            extreme_df.loc[idx, 'sleep_duration_sum'] /= 2  

       
        health_dates = np.random.choice(unique_dates, size=5, replace=False)
        for date in health_dates:
            idx = extreme_df[extreme_df['Date'] == date].index
            extreme_df.loc[idx, 'bp_systolic_mean'] -= 10  
            extreme_df.loc[idx, 'hrv_mean'] += 15         
            extreme_df.loc[idx, 'sleep_duration_sum'] *= 1.2 

        final_injected_df.append(pd.concat([user_df, extreme_df], ignore_index=True).drop_duplicates(subset=['Date', 'user_id'], keep='first'))

    return pd.concat(final_injected_df, ignore_index=True)


def align_mood_sentiment(df):
    """
    CRITICAL STEP: Aligns the synthesized mood log with the underlying physiological stress rank,
    calculated and applied PER USER.
    """
    logging.info("Causally aligning mood sentiment with physiological data...")

    final_aligned_df = []
    
    for user_id in df['user_id'].unique():
        user_df = df[df['user_id'] == user_id].copy()
        
        df_temp = user_df.copy()
        
       
        def calculate_z_score_internal(series):
            std_dev = series.std()
            if std_dev < EPSILON: 
                return pd.Series(0.0, index=series.index)
            return (series - series.mean()) / std_dev 

        df_temp['bp_sys_z'] = calculate_z_score_internal(df_temp['bp_systolic_mean'])
        df_temp['hrv_z'] = calculate_z_score_internal(df_temp['hrv_mean'])
        df_temp['sleep_z'] = calculate_z_score_internal(df_temp['sleep_duration_sum'])
        
        df_temp['Stress_Rank'] = (df_temp['bp_sys_z'] * 0.3) + (-df_temp['hrv_z'] * 0.4) + (-df_temp['sleep_z'] * 0.3)

       
        q_low = df_temp['Stress_Rank'].quantile(0.25)
        q_high = df_temp['Stress_Rank'].quantile(0.75)
        
        mood_log_list = []
        sentiment_list = []

        for rank in df_temp['Stress_Rank']:
            if rank < q_low:
                mood = np.random.choice(MOOD_LOGS['POSITIVE'])
            elif rank > q_high:
                mood = np.random.choice(MOOD_LOGS['NEGATIVE'])
            else:
                mood = np.random.choice(MOOD_LOGS['NEUTRAL'])
            
            mood_log_list.append(mood)
            sentiment_list.append(get_sentiment_score(mood))
            
        user_df['mood_log_text'] = mood_log_list
        user_df['sentiment_compound_mean'] = sentiment_list
        
        final_aligned_df.append(user_df)
        
    return pd.concat(final_aligned_df, ignore_index=True)


def feature_engineering(df):
    """
    Calculates lagged features, crucial for time-series prediction.
    Must be performed PER USER.
    """
    
    features_to_lag = config.CORE_FEATURE_BASES
    
    def create_lags(user_group):
        user_group = user_group.sort_values(by='Date')
        for feature in features_to_lag:
            for lag in range(1, 4): 
                user_group[f'{feature}_lag_{lag}'] = user_group[feature].shift(lag)
        return user_group

    df = df.groupby('user_id', group_keys=False).apply(create_lags)
    df.dropna(inplace=True)
    return df

def robust_z_score(series, name):
    """
    Calculates Z-score. Fixed to prevent recursive call error by using standard 
    math operation instead of internal function call for STD.
    """
    
    std_dev = series.std()
    
    if pd.isna(std_dev) or std_dev < EPSILON:
        logging.warning(f"Standard deviation for '{name}' is zero or near-zero ({std_dev:.2e}). Z-score set to 0.0.")
        return pd.Series(0.0, index=series.index)
    else:
        
        return (series - series.mean()) / std_dev 

def derive_mental_health_labels(df):
    """
    Synthesizes GAD-7 and Stress Level labels using a HYBRID score, performed PER USER.
    """
    if df.empty:
        logging.error("DataFrame is empty, cannot synthesize labels.")
        return df

    logging.info("Starting HYBRID GAD-7 synthesis (Relative + Absolute Risk)...")
    
    final_labeled_df = []
    
    for user_id in df['user_id'].unique():
        user_df = df[df['user_id'] == user_id].copy()

        
        user_df['steps_z'] = robust_z_score(user_df['steps_sum'], 'steps_sum')
        user_df['sleep_z'] = robust_z_score(user_df['sleep_duration_sum'], 'sleep_duration_sum')
        user_df['audio_std_z'] = robust_z_score(user_df['audio_level_std'], 'audio_level_std')
        user_df['bp_sys_z'] = robust_z_score(user_df['bp_systolic_mean'], 'bp_systolic_mean')
        user_df['hrv_z'] = robust_z_score(user_df['hrv_mean'], 'hrv_mean')
        user_df['sentiment_z'] = robust_z_score(user_df['sentiment_compound_mean'], 'sentiment_compound_mean')

        relative_stress_score = (
            (-user_df['steps_z'] * 0.05) + (-user_df['sleep_z'] * 0.10) + (user_df['audio_std_z'] * 0.05) +
            (user_df['bp_sys_z'] * 0.15) + (-user_df['hrv_z'] * 0.15) + (-user_df['sentiment_z'] * 0.25) 
        )

        
        absolute_risk_score = pd.Series(0.0, index=user_df.index)
        chronic_means = user_df[[c for c in user_df.columns if c.endswith('_mean') or c.endswith('_sum')]].mean()
        
        for feature, target in config.INDUSTRY_TARGETS.items():
            if feature in chronic_means:
                deviation = abs(chronic_means[feature] - target)
                scale_factor = target * 0.20 
                risk_contribution = np.clip(deviation / scale_factor, 0, 1.0)
                absolute_risk_score += risk_contribution * (config.ABSOLUTE_RISK_WEIGHT / len(config.INDUSTRY_TARGETS))
        
        absolute_risk_score_value = absolute_risk_score.mean()
        
        
        raw_stress_score = relative_stress_score + absolute_risk_score_value + (np.random.normal(0, 0.1, len(user_df)))
        
        min_score = raw_stress_score.min()
        max_score = raw_stress_score.max()
        
        if (max_score - min_score) < EPSILON:
            user_df['Stress_Level_Raw'] = 0.5 
        else:
            user_df['Stress_Level_Raw'] = (raw_stress_score - min_score) / (max_score - min_score)
        
        user_df['GAD7_Score'] = np.clip(np.round(user_df['Stress_Level_Raw'] * 21).astype(int), 0, 21)
        
        user_df['Anxiety_Severity'] = pd.cut(
            user_df['GAD7_Score'],
            bins=[-1, 4, 9, 14, 21],
            labels=['Minimal', 'Mild', 'Moderate', 'Severe'],
            right=True, ordered=True
        ).astype('category')

        drop_cols = [col for col in user_df.columns if col.endswith('_z') or col.endswith('_Raw')]
        user_df.drop(columns=drop_cols, inplace=True)
        final_labeled_df.append(user_df)

    return pd.concat(final_labeled_df, ignore_index=True)

if __name__ == '__main__':
    if not os.path.exists(RAW_DATA_DIR):
        logging.error(f"Directory '{RAW_DATA_DIR}' not found.")
        
    final_combined_df = load_and_combine_data(None, None) 
    
    if final_combined_df is not None and not final_combined_df.empty:
        
        
        extreme_df = inject_extreme_periods(final_combined_df) 

        
        aligned_df = align_mood_sentiment(extreme_df)

        
        processed_df = feature_engineering(aligned_df)

        if not processed_df.empty:
            
            final_df = derive_mental_health_labels(processed_df)

            
            if not final_df.empty:
                final_df.to_csv(OUTPUT_FILE, index=False)
                logging.info(f"\n--- Data Preprocessing Complete ---")
                logging.info(f"Shape of the final dataset: {final_df.shape}")
                logging.info(f"Synthesized GAD-7 Score distribution:\n{final_df['Anxiety_Severity'].value_counts()}")
            else:
                logging.error("Final DataFrame is empty after label synthesis.")
        else:
            logging.error("DataFrame is empty after feature engineering (lagging).")
    else:
        logging.error("Initial data loading failed.")

