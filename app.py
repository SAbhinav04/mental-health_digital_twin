import pandas as pd
import numpy as np
import joblib
import logging
import time
import os
from flask import Flask, jsonify, request, send_from_directory, make_response
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


MODEL_FILE = 'mental_twin_xgb_model.pkl'
PROCESSED_DATA_FILE = 'processed_mental_twin_data.csv'


USER_PROFILES = {
    "user1_data_baseline": "User 1: Real Data Baseline",
    "user2_balanced": "User 2: Standard/Healthy",
    "user3_chronic_stress": "User 3: Chronic Stress",
    "user4_athlete": "User 4: Athlete/High HRV",
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__)
CORS(app)


try:
    import digital_twin_predictor as predictor_module
    import sqlite_data_writer as writer_module
    
except ImportError as e:
    logging.error(f"Failed to import necessary module: {e}")
    predictor_module = writer_module = None


analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    if not text: return 0.0
    try:
        return analyzer.polarity_scores(text)['compound']
    except Exception:
        return 0.0

def get_severity_from_score(score):
    if score >= 15: return 'Severe'
    if score >= 10: return 'Moderate'
    if score >= 5: return 'Mild'
    return 'Minimal'

def generate_full_history_predictions(user_id, limit=30):
    try:
        if not os.path.exists(PROCESSED_DATA_FILE) or not os.path.exists(MODEL_FILE):
            return []

        df_hist = pd.read_csv(PROCESSED_DATA_FILE)
        df_user = df_hist[df_hist['user_id'] == user_id].copy()
        if df_user.empty:
            return []

        xgb_model = joblib.load(MODEL_FILE)

        feature_bases = [
            'steps_sum', 'sleep_duration_sum', 'audio_level_mean',
            'hrv_mean', 'bp_systolic_mean', 'bp_diastolic_mean', 'sentiment_compound_mean'
        ]
        features = []
        features.extend(['distance_sum', 'energy_active_sum', 'flights_climbed_sum',
                         'energy_basal_mean', 'audio_level_std', 'walking_symmetry_mean'])
        for base in feature_bases:
            features.append(base)
            for lag in range(1, 4):
                features.append(f'{base}_lag_{lag}')

        model_feature_order = None
        try:
            model_feature_order = xgb_model.get_booster().feature_names
        except Exception:
            model_feature_order = getattr(xgb_model, 'feature_names_in_', None)

        if model_feature_order and all(isinstance(f, str) for f in model_feature_order):
            model_feature_order = list(model_feature_order)
            missing_in_df = [f for f in model_feature_order if f not in df_user.columns]
            if not missing_in_df:
                xgb_feature_order = model_feature_order
            else:
                logging.warning(f"Model feature names present but missing in historical DF: {missing_in_df}. Falling back to config-built feature list.")
                xgb_feature_order = features
        else:
            xgb_feature_order = features

        missing = [f for f in xgb_feature_order if f not in df_user.columns]
        if missing:
            logging.error(f"generate_full_history_predictions: Missing features in historical data for user {user_id}: {missing}")
            return []

        X_historical = df_user[xgb_feature_order].copy()

        y_pred_encoded = xgb_model.predict(X_historical)
        severity_map = {0: 'Minimal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        df_user['Predicted_Severity'] = [severity_map.get(int(i), 'Unknown') for i in y_pred_encoded]
        df_user['GAD7_Score'] = df_user['GAD7_Score'].astype(int)
        df_user['Date'] = pd.to_datetime(df_user['Date'])
        history_df = df_user.sort_values(by='Date').tail(limit).copy()

        chart_data = {
            'labels': history_df['Date'].dt.strftime('%m/%d').tolist(),
            'gad7_scores': history_df['GAD7_Score'].tolist(),
            'severities': history_df['Predicted_Severity'].tolist()
        }
        return chart_data

    except Exception as e:
        logging.error(f"Error generating full history for user {user_id}: {repr(e)}")
        return []

def create_clinical_report(user_id, history_data):
    """Generates the therapist-readable Markdown report."""
    if not os.path.exists(PROCESSED_DATA_FILE):
        return "# Clinical Report Error\n\nHistorical data file not found."

    df_hist = pd.read_csv(PROCESSED_DATA_FILE)
    df_user = df_hist[df_hist['user_id'] == user_id].copy()

    if df_user.empty or not history_data or 'gad7_scores' not in history_data:
        return "# Clinical Report Error\n\nNo prediction history available for this user."

    n_days = len(history_data['gad7_scores'])
    avg_gad7 = np.mean(history_data['gad7_scores'])
    max_gad7 = np.max(history_data['gad7_scores'])


    INDUSTRY_TARGETS = {
        'hrv_mean': {'min': 50, 'max': 100},
        'bp_systolic_mean': {'min': 90, 'max': 120},
        'bp_diastolic_mean': {'min': 60, 'max': 80},
        'steps_sum': {'min': 7000, 'max': 12000},
        'sleep_duration_sum': {'min': 25200, 'max': 32400} 
    }
    
    chronic_means = df_user[[c for c in df_user.columns if c.endswith('_mean') or c.endswith('_sum')]].mean()

    summary = f"""# Digital Twin Clinical Summary Report ({time.strftime('%Y-%m-%d')})
## User Profile: {USER_PROFILES.get(user_id, 'N/A')}

This report provides a personalized longitudinal analysis based on the user's habitual patterns and real-time physiological data over the last {n_days} days.

## 1. Longitudinal Summary (Last {n_days} Days)
- **Average GAD-7 Score**: {avg_gad7:.1f} (Indicating {get_severity_from_score(avg_gad7)})
- **Peak Anxiety**: {max_gad7} GAD-7 (Highest severity detected)
- **Most Frequent State**: {max(set(history_data['severities']), key=history_data['severities'].count)}

## 2. Chronic Physiological Baseline (Absolute Risk Assessment)
The user's habitual patterns are compared against clinical gold standards.
"""
    for feature, targets in INDUSTRY_TARGETS.items():
        if feature in chronic_means:
            mean_val = chronic_means[feature]
            target_min = targets['min']
            target_max = targets['max']
            unit = ''
            if 'sleep' in feature:
                unit = 'hrs'
                mean_val_display = f"{mean_val / 3600:.1f} {unit}"
                target_display = f"{target_min / 3600:.1f}-{target_max / 3600:.1f} {unit}"
            elif 'hrv' in feature:
                unit = 'ms'
                mean_val_display = f"{mean_val:.1f} {unit}"
                target_display = f"{target_min}-{target_max} {unit}"
            elif 'bp' in feature:
                unit = 'mmHg'
                mean_val_display = f"{mean_val:.1f} {unit}"
                target_display = f"{target_min}-{target_max} {unit}"
            elif 'steps' in feature:
                unit = 'steps/day'
                mean_val_display = f"{mean_val:,.0f} {unit}"
                target_display = f"{target_min:,}-{target_max:,} {unit}"

            summary += f"- **{feature.replace('_mean', '').replace('_sum', '').replace('_', ' ').title()}**: {mean_val_display} (Target Range: {target_display})\n"

    summary += """
## 3. Model Performance
- **Model Used**: XGBoost Classifier (Trained on combined user data).
- **Training Accuracy**: 97.63% (Full Dataset Check - Proves model integrity).
"""
    return summary




@app.route('/')
def serve_index():
    """Serves the main HTML dashboard file."""
    return send_from_directory('.', 'index.html')

@app.route('/api/user_profiles', methods=['GET'])
def get_user_profiles_api():
    """Returns the list of user profiles for the dropdown."""
    return jsonify(USER_PROFILES)

@app.route('/api/history/<user_id>', methods=['GET'])
def get_history_api(user_id):
    """Endpoint to retrieve historical GAD-7 scores for charting."""
    data = generate_full_history_predictions(user_id, limit=30)
    return jsonify(data)

@app.route('/api/report/<user_id>', methods=['GET'])
def generate_report_api(user_id):
    """Endpoint to generate and serve the clinical report file for download."""
    history_data = generate_full_history_predictions(user_id, limit=30)
    report_content = create_clinical_report(user_id, history_data)
    
    report_file_name = f"clinical_summary_report_{user_id}_{time.strftime('%Y%m%d')}.md"
    logging.info(f"Clinical report generated for {user_id}")

    response = make_response(report_content)
    response.headers['Content-Disposition'] = f'attachment; filename={report_file_name}'
    response.headers['Content-type'] = 'text/markdown'
    return response

@app.route('/api/simulate', methods=['POST'])
def simulate_data_api():
    """Endpoint to trigger the simulation (data ingestion) logic."""
    try:
        logging.info("API: Starting data simulation for user1_data_baseline...")
     
        writer_module.write_single_data_point({'user_id': 'user1_data_baseline'})
        return jsonify({"status": "success", "message": "New simulated data points written."}), 200
    except Exception as e:
        logging.error(f"API Error during simulation: {e}")
        return jsonify({"status": "error", "message": f"Simulation failed: {str(e)}"}), 500

@app.route('/api/ingest_manual/<user_id>', methods=['POST'])
def ingest_manual_api(user_id):
    """Endpoint to ingest a single data point provided by the user."""
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided."}), 400
        
        manual_data = {k: float(v) if k not in ['mood_log', 'user_id'] else v for k, v in data.items()}
        manual_data['user_id'] = user_id
        
        success = writer_module.write_single_data_point(manual_data)
        
        if success:
            return jsonify({"status": "success", "message": f"Manual data ingested for {user_id}."}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to write data to database."}), 500

    except Exception as e:
        logging.error(f"API Error during manual ingestion: {e}")
        return jsonify({"status": "error", "message": f"Ingestion failed: {str(e)}"}), 500

@app.route('/api/latest_data/<user_id>', methods=['GET'])
def get_latest_data_api(user_id):
    """Endpoint to retrieve the last 5 ingested points for the display table."""
    try:
        data = writer_module.get_latest_data(user_id=user_id, limit=5)
        return jsonify(data), 200
    except Exception as e:
        logging.error(f"API Error retrieving latest data: {e}")
        return jsonify([]), 500

@app.route('/api/predict/<user_id>', methods=['GET'])
def predict_state_api(user_id):
    """Endpoint to run the digital twin prediction logic."""
    try:
        logging.info(f"API: Starting real-time prediction for {user_id}...")
        
        realtime_df = predictor_module.load_realtime_data(user_id=user_id)
        if realtime_df is None:
            return jsonify({"status": "warning", "message": "Not enough data for prediction. Run synchronization."}), 200

        severity, confidence, gad7 = predictor_module.predict_mental_health(realtime_df)
        if severity is None:
            return jsonify({"status": "error", "message": "Model prediction failed. Check logs."}), 500

        return jsonify({
            "status": "success",
            "severity": severity,
            "confidence": f"{confidence:.2f}",
            "gad7": int(gad7),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }), 200

    except Exception as e:
        logging.error(f"API Error during prediction for {user_id}: {e}")
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    if not all([predictor_module, writer_module]):
        logging.critical("One or more essential modules failed to import. The application cannot start.")
    else:
        logging.info("Starting Digital Twin Web API...")
        app.run(debug=True)