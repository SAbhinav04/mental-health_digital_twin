"""
FastAPI application for Mental Health Digital Twin.

Provides REST API for:
- Real-time mental health predictions
- Historical data tracking
- Clinical reports
- Model explainability (SHAP)
- Feature drift monitoring

Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import os
import platform

# Fix for multiple OpenMP runtimes crashing the process
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Apple Silicon (M5) threading optimization
if platform.system() == "Darwin" and platform.machine() == "arm64":
    # Force single-threaded loading to bypass OpenMP initialization bugs
    os.environ['OMP_NUM_THREADS'] = '1'
    # Prevents XGBoost from conflicting with the Python 3.14 signal handlers
    os.environ['XGBOOST_CURRENT_THREAD_CHECK'] = '0'

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import project modules
import config
try:
    import digital_twin_predictor as predictor_module
    import sqlite_data_writer as writer_module
    MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Failed to import essential modules: {e}")
    MODULES_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    status: str
    severity: Optional[str] = None
    confidence: Optional[float] = None
    gad7: Optional[int] = None
    timestamp: str
    message: Optional[str] = None
    shap_explanations: Optional[List[Dict[str, Any]]] = None
    explanation_summary: Optional[str] = None


class SimulationResponse(BaseModel):
    """Response model for simulation."""
    status: str
    message: str


class HistoryPoint(BaseModel):
    """Single history data point."""
    date: str
    gad7_score: int
    severity: str


class HistoryResponse(BaseModel):
    """Response model for history."""
    user_id: str
    labels: List[str]
    gad7_scores: List[int]
    severities: List[str]


class DataPoint(BaseModel):
    """Manual data point ingestion."""
    user_id: str
    hrv: Optional[float] = None
    sleep_duration: Optional[float] = None
    steps: Optional[int] = None
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    audio_level: Optional[float] = None
    mood_log: Optional[str] = None


class DriftReport(BaseModel):
    """Drift detection report."""
    status: str
    timestamp: str
    drifted_features_count: int
    drift_percentage: float
    summary: str
    report_path: Optional[str] = None


class ExplainResponse(BaseModel):
    """SHAP explanation response."""
    user_id: str
    severity: Optional[str] = None
    confidence: Optional[float] = None
    shap_explanations: List[Dict[str, Any]]
    summary: str


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Mental Health Digital Twin API",
    description="REST API for mental health prediction and monitoring",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (index.html)
if os.path.exists('index.html'):
    app.mount("/static", StaticFiles(directory="."), name="static")


# ============================================================================
# Helper Functions
# ============================================================================

def get_severity_from_score(score: float) -> str:
    """Map GAD-7 score to severity label."""
    if score >= 15:
        return 'Severe'
    if score >= 10:
        return 'Moderate'
    if score >= 5:
        return 'Mild'
    return 'Minimal'


def generate_full_history_predictions(user_id: str, limit: int = 30) -> Optional[Dict]:
    """Generate prediction history for a user."""
    try:
        PROCESSED_DATA_FILE = 'processed_mental_twin_data.csv'
        MODEL_FILE = 'mental_twin_xgb_model.pkl'
        
        if not os.path.exists(PROCESSED_DATA_FILE) or not os.path.exists(MODEL_FILE):
            return None

        import joblib
        df_hist = pd.read_csv(PROCESSED_DATA_FILE)
        df_user = df_hist[df_hist['user_id'] == user_id].copy()
        
        if df_user.empty:
            return None

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

        model_feature_order = features
        missing = [f for f in model_feature_order if f not in df_user.columns]
        
        if missing:
            return None

        X_historical = df_user[model_feature_order].copy()

        y_pred_encoded = xgb_model.predict(X_historical)
        severity_map = {0: 'Minimal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        df_user['Predicted_Severity'] = [severity_map.get(int(i), 'Unknown') for i in y_pred_encoded]
        df_user['GAD7_Score'] = df_user['GAD7_Score'].astype(int)
        df_user['Date'] = pd.to_datetime(df_user['Date'])
        history_df = df_user.sort_values(by='Date').tail(limit).copy()

        chart_data = {
            'user_id': user_id,
            'labels': history_df['Date'].dt.strftime('%m/%d').tolist(),
            'gad7_scores': history_df['GAD7_Score'].tolist(),
            'severities': history_df['Predicted_Severity'].tolist()
        }
        return chart_data

    except Exception as e:
        logging.error(f"Error generating full history for user {user_id}: {repr(e)}")
        return None


def create_clinical_report(user_id: str, history_data: Optional[Dict]) -> str:
    """Generate Markdown clinical report."""
    if history_data is None or not history_data.get('gad7_scores'):
        return "# Clinical Report Error\n\nNo prediction history available for this user."

    n_days = len(history_data['gad7_scores'])
    avg_gad7 = np.mean(history_data['gad7_scores'])
    max_gad7 = np.max(history_data['gad7_scores'])

    report = f"""# Digital Twin Clinical Summary Report ({time.strftime('%Y-%m-%d')})

## User: {user_id}

This report provides a personalized longitudinal analysis based on the user's habitual patterns and real-time physiological data.

## Longitudinal Summary (Last {n_days} Days)
- **Average GAD-7 Score**: {avg_gad7:.1f} (Indicating {get_severity_from_score(avg_gad7)})
- **Peak Anxiety**: {max_gad7} GAD-7 (Highest severity detected)
- **Most Frequent State**: {max(set(history_data['severities']), key=history_data['severities'].count)}

## Clinical Model
- **Model Used**: XGBoost Classifier with temporal features
- **Classes**: Minimal (0-4), Mild (5-9), Moderate (10-14), Severe (15+)
- **Features**: Physiological (HRV, BP, Sleep, Steps) + NLP sentiment

## ⚠️ Clinical Disclaimer
**This tool is not a diagnostic instrument.** Predictions are for monitoring purposes only and should be reviewed by a qualified mental health professional. Always consult with healthcare providers for clinical decision-making.

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return report


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve the main dashboard."""
    if os.path.exists('index.html'):
        return FileResponse('index.html', media_type='text/html')
    return {"message": "Mental Health Digital Twin API v2.0"}


@app.get("/api/user_profiles")
async def get_user_profiles() -> Dict[str, Any]:
    """Get available user profiles for dropdown. Returns profile metadata dicts."""
    return config.USER_PROFILES


@app.get("/api/history/{user_id}")
async def get_history(user_id: str) -> HistoryResponse:
    """Get prediction history for a user."""
    data = generate_full_history_predictions(user_id, limit=30)
    
    if data is None:
        raise HTTPException(status_code=404, detail="No history found for user")
    
    return HistoryResponse(
        user_id=user_id,
        labels=data['labels'],
        gad7_scores=data['gad7_scores'],
        severities=data['severities']
    )


@app.get("/api/report/{user_id}")
async def generate_report(user_id: str):
    """Generate and return clinical report as Markdown."""
    history_data = generate_full_history_predictions(user_id, limit=30)
    report_content = create_clinical_report(user_id, history_data)
    
    filename = f"clinical_report_{user_id}_{time.strftime('%Y%m%d')}.md"
    
    return StreamingResponse(
        iter([report_content]),
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/api/simulate")
async def simulate_data() -> SimulationResponse:
    """Trigger data simulation."""
    if not MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Modules not available")
    
    try:
        logging.info("API: Starting data simulation for user1_data_baseline...")
        writer_module.write_single_data_point({'user_id': 'user1_data_baseline'})
        
        return SimulationResponse(
            status="success",
            message="New simulated data points written."
        )
    except Exception as e:
        logging.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@app.post("/api/ingest_manual/{user_id}")
async def ingest_manual_data(user_id: str, data: DataPoint) -> SimulationResponse:
    """Ingest a single manual data point."""
    if not MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Modules not available")
    
    try:
        manual_data = {
            'user_id': user_id,
            'hrv': data.hrv,
            'sleep_duration': data.sleep_duration,
            'steps': data.steps,
            'bp_systolic': data.bp_systolic,
            'bp_diastolic': data.bp_diastolic,
            'audio_level': data.audio_level,
            'mood_log': data.mood_log or ''
        }
        
        success = writer_module.write_single_data_point(manual_data)
        
        if success:
            return SimulationResponse(
                status="success",
                message=f"Manual data ingested for {user_id}."
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to write data")
    
    except Exception as e:
        logging.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/api/predict/{user_id}")
async def predict_state(user_id: str) -> PredictionResponse:
    """Make a mental health prediction for a user."""
    if not MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Modules not available")
    
    try:
        logging.info(f"API: Predicting for {user_id}...")
        
        realtime_df = predictor_module.load_realtime_data(user_id=user_id)
        if realtime_df is None:
            return PredictionResponse(
                status="warning",
                message="Not enough data for prediction. Run simulation first.",
                timestamp=datetime.now().isoformat()
            )

        # Get prediction with SHAP explanation
        result = predictor_module.predict_with_explanation(realtime_df, user_id, use_shap=True)
        
        return PredictionResponse(
            status="success",
            severity=result['severity'],
            confidence=result['confidence'],
            gad7=result['gad7'],
            timestamp=result['timestamp'],
            shap_explanations=result.get('shap_explanations'),
            explanation_summary=result.get('explanation_summary')
        )

    except Exception as e:
        logging.error(f"Prediction error for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/explain/{user_id}")
async def get_explanation(user_id: str) -> ExplainResponse:
    """Get SHAP explanation for latest prediction."""
    if not MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Modules not available")
    
    try:
        realtime_df = predictor_module.load_realtime_data(user_id=user_id)
        if realtime_df is None:
            raise HTTPException(status_code=404, detail="No data for user")
        
        result = predictor_module.predict_with_explanation(realtime_df, user_id, use_shap=True)
        
        return ExplainResponse(
            user_id=user_id,
            severity=result['severity'],
            confidence=result['confidence'],
            shap_explanations=result.get('shap_explanations', []),
            summary=result.get('explanation_summary', 'No summary available')
        )
    
    except Exception as e:
        logging.error(f"Explanation error for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/api/drift_report/{user_id}")
async def drift_report(user_id: str, days: int = 7) -> DriftReport:
    """Generate drift detection report."""
    try:
        from drift_monitor import DriftMonitor
        
        # Load baseline (training data)
        try:
            baseline_df = pd.read_csv('processed_mental_twin_data.csv')
        except Exception as e:
            logging.warning(f"Could not load baseline: {e}")
            raise HTTPException(status_code=503, detail="Baseline data not available")
        
        monitor = DriftMonitor(baseline_df=baseline_df)
        drift_status = monitor.get_drift_status(user_id, days=days)
        
        # Generate HTML report
        report_path = monitor.generate_report(user_id, days=days)
        
        return DriftReport(
            status=drift_status.get('status', 'unknown'),
            timestamp=datetime.now().isoformat(),
            drifted_features_count=drift_status.get('drifted_features_count', 0),
            drift_percentage=drift_status.get('drift_percentage', 0),
            summary=drift_status.get('summary', ''),
            report_path=report_path
        )
    
    except Exception as e:
        logging.error(f"Drift report error for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@app.get("/api/latest_data/{user_id}")
async def get_latest_data(user_id: str, limit: int = 5) -> List[Dict]:
    """Get latest ingested data points for a user."""
    if not MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Modules not available")
    
    try:
        data = writer_module.get_latest_data(user_id=user_id, limit=limit)
        return data
    except Exception as e:
        logging.error(f"Error retrieving latest data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve data")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modules_available": str(MODULES_AVAILABLE)
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    if not MODULES_AVAILABLE:
        logging.warning("Some modules are not available. API will have limited functionality.")
    
    logging.info("Starting Mental Health Digital Twin API (FastAPI)...")
    logging.info("Access the API at: http://localhost:8000")
    logging.info("API Documentation at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
