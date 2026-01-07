import sqlite3
import pandas as pd
import joblib
import numpy as np
import logging
import time
import random 
import os 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import config 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_NAME = 'realtime_data.db'
MODEL_FILE = 'mental_twin_xgb_model.pkl'
PROCESSED_DATA_FILE = 'processed_mental_twin_data.csv'
analyzer = SentimentIntensityAnalyzer() 

from transformers import pipeline

# Load once (global)
bert_sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

def get_sentiment_score(text):
    """
    Returns a sentiment score in the range [-1, 1],
    compatible with existing model features.
    """
    if not text:
        return 0.0

    try:
        result = bert_sentiment(text[:512])[0]
        label = result['label']
        confidence = result['score']

        # Map labels to [-1, 0, +1]
        if label == 'LABEL_2':      # Positive
            return confidence
        elif label == 'LABEL_0':    # Negative
            return -confidence
        else:                       # Neutral
            return 0.0

    except Exception as e:
        logging.warning(f"BERT sentiment failed, falling back to VADER: {e}")
        return analyzer.polarity_scores(text)['compound']


def load_realtime_data(user_id='user1_data_baseline'):
    """
    Loads real-time data, aggregates it, and assembles the final 34-feature vector 
    by retrieving components from the historical CSV file, filtered by user.
    """
    
    logging.info(f"Reading real-time data from SQLite for {user_id}...")
    
    conn = sqlite3.connect(DATABASE_NAME)
    df_raw = pd.read_sql_query(f"SELECT * FROM time_series WHERE user_id = ? ORDER BY timestamp DESC;", conn, params=(user_id,))
    conn.close()

    if df_raw.empty:
        logging.error(f"No recent data found in the SQLite database for {user_id}.")
        return None

  
    latest_mood_text = df_raw.iloc[0]['mood_log'] if 'mood_log' in df_raw.columns else ""
    current_sentiment_score = get_sentiment_score(latest_mood_text)

    
    df_raw.rename(columns={
        'steps': 'steps_sum', 'sleep_duration': 'sleep_duration_sum',
        'audio_level': 'audio_level_mean', 'hrv': 'hrv_mean',
        'bp_systolic': 'bp_systolic_mean', 'bp_diastolic': 'bp_diastolic_mean'
    }, inplace=True)
    

    features_to_predict_realtime = [
        'steps_sum', 'sleep_duration_sum', 'audio_level_mean', 
        'hrv_mean', 'bp_systolic_mean', 'bp_diastolic_mean'
    ]
    
    current_prediction_row = df_raw[features_to_predict_realtime].mean().to_frame().T 

    current_prediction_row['sentiment_compound_mean'] = current_sentiment_score


    try:
        if not os.path.exists(PROCESSED_DATA_FILE):
             logging.error(f"Historical file '{PROCESSED_DATA_FILE}' not found. Cannot retrieve lag features.")
             return None

        df_hist = pd.read_csv(PROCESSED_DATA_FILE)
        
        
        df_hist_user = df_hist[df_hist['user_id'] == user_id].copy()
        if df_hist_user.empty:
            logging.error(f"Historical data file is missing data for user {user_id}.")
            return None
             
        
        last_hist_row = df_hist_user.iloc[-1]
        
        
        final_prediction_row = current_prediction_row.copy()
        
        
        non_lagged_summary_bases = config.NON_LAGGED_SUMMARY_BASES

        
        for base in non_lagged_summary_bases:
            
            final_prediction_row[base] = last_hist_row[base]
        
       
        lag_bases = config.CORE_FEATURE_BASES
        
       
        for feature_base in lag_bases:
            for lag in range(1, 4):
                col_name = f'{feature_base}_lag_{lag}'
               
                last_row_value = last_hist_row.get(col_name)
                final_prediction_row[col_name] = last_row_value if not pd.isna(last_row_value) else 0.0 
        
        
        logging.info(f"DIAGNOSTIC: Features assembled. Count: {len(final_prediction_row.columns)}")
        if len(final_prediction_row.columns) != 34:
             logging.error(f"FATAL SYNC ERROR: Expected 34 features, got {len(final_prediction_row.columns)}. Columns: {final_prediction_row.columns.tolist()}")
             return None

        return final_prediction_row

    except Exception as e:
        logging.error(f"Prediction data loading failed: {e}")
        return None

# def predict_mental_health(data):
#     """Loads the model and predicts the Anxiety_Severity for the new data point."""
#     try:
#         if not os.path.exists(MODEL_FILE):
#             logging.error(f"Model file '{MODEL_FILE}' not found. Did you run digital_twin_model.py?")
#             return None, None, None

#         xgb_model = joblib.load(MODEL_FILE)
#         logging.info(f"Loaded trained Digital Twin model from mental_twin_xgb_model.pkl")

#         # 1. Ensure features match the 34 columns the model expects
#         feature_names = xgb_model.get_booster().feature_names
#         X_predict = data[feature_names] 
        
#         # 2. Make Prediction
#         prediction_int = xgb_model.predict(X_predict)[0]
        
#         # 3. Map prediction back to severity name
#         severity_map = {0: 'Minimal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
#         predicted_severity = severity_map.get(prediction_int, "Unknown")
        
#         # 4. Confidence and GAD-7 Score
#         prediction_proba = xgb_model.predict_proba(X_predict)[0]
#         confidence = prediction_proba[prediction_int] * 100
        
#         simulated_gad7 = np.clip(random.randint(int(prediction_int * 5 - 1), int(prediction_int * 5 + 3)), 0, 21)

#         return predicted_severity, confidence, simulated_gad7

#     except Exception as e:
#         logging.error(f"Prediction failed: {e}")
#         return None, None, None

def predict_mental_health(data):
    """Loads the model and predicts the Anxiety_Severity for the new data point."""
    try:
        if not os.path.exists(MODEL_FILE):
            logging.error(f"Model file '{MODEL_FILE}' not found. Did you run digital_twin_model.py?")
            return None, None, None

        xgb_model = joblib.load(MODEL_FILE)
        logging.info(f"Loaded trained Digital Twin model from {MODEL_FILE}")

        
        model_feature_names = None
        try:
            model_feature_names = xgb_model.get_booster().feature_names
        except Exception:
            model_feature_names = getattr(xgb_model, 'feature_names_in_', None)

        if not model_feature_names or not all(isinstance(f, str) for f in model_feature_names):
            
            feature_bases = config.CORE_FEATURE_BASES
            non_lagged_summary_bases = config.NON_LAGGED_SUMMARY_BASES
            features = []
            features.extend(non_lagged_summary_bases)
            for base in feature_bases:
                features.append(base)
                for lag in range(1, 4):
                    features.append(f'{base}_lag_{lag}')
            model_feature_names = features

        
        X_predict = data.copy()
        if isinstance(X_predict, pd.Series):
            X_predict = X_predict.to_frame().T
        if not isinstance(X_predict, pd.DataFrame):
            X_predict = pd.DataFrame(X_predict)

        
        missing = [f for f in model_feature_names if f not in X_predict.columns]
        if missing:
            raise KeyError(f"Missing features for prediction: {missing}")

        
        X_predict = X_predict[model_feature_names].astype(float)

        
        prediction_int = int(xgb_model.predict(X_predict)[0])
        prediction_proba = xgb_model.predict_proba(X_predict)[0]
        confidence = float(prediction_proba[prediction_int]) * 100.0

        severity_map = {0: 'Minimal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        predicted_severity = severity_map.get(prediction_int, "Unknown")

        
        simulated_gad7 = int(np.clip(random.randint(int(prediction_int * 5 - 1), int(prediction_int * 5 + 3)), 0, 21))

        return predicted_severity, confidence, simulated_gad7

    except Exception as e:
        logging.error(f"Prediction failed: {repr(e)}")
        return None, None, None
    
def predict_from_daic_woz_dialogue(utterances, user_id='user1_data_baseline'):
    """
    Simulates DAIC-WoZ dialogue integration by aggregating
    multiple conversational utterances into one NLP signal.
    """
    full_text = " ".join(utterances)
    return predict_from_text_only(full_text, user_id)

def predict_from_text_only(text, user_id='user1_data_baseline'):
    logging.info("Running TEXT-ONLY prediction pipeline...")

    sentiment_score = get_sentiment_score(text)

    try:
        data = load_realtime_data(user_id)
    except Exception:
        logging.warning("Realtime DB not available. Using historical-only fallback.")
        data = None

    if data is None:
        df_hist = pd.read_csv(PROCESSED_DATA_FILE)
        df_hist_user = df_hist[df_hist['user_id'] == user_id]

        if df_hist_user.empty:
            logging.error("No historical data available for fallback.")
            return None, None, None

        data = df_hist_user.iloc[-1:].copy()

    data['sentiment_compound_mean'] = sentiment_score

    return predict_mental_health(data)


from daic_woz_sample import DAIC_WOZ_SAMPLE_UTTERANCES

if __name__ == '__main__':
    severity, confidence, gad7 = predict_from_daic_woz_dialogue(
        DAIC_WOZ_SAMPLE_UTTERANCES
    )

    print("DAIC-WoZ Integration Test")
    print("Severity:", severity)
    print("Confidence:", confidence)
    print("GAD-7:", gad7)
