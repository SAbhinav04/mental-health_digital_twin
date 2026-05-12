# Mental Health Digital Twin - v2.0 Implementation Guide

## Overview

This document outlines all improvements implemented for the Mental Health Digital Twin system. The project now includes real dataset support, temporal modeling, explainability, rigorous evaluation, drift monitoring, and production-ready infrastructure.

---

## 📋 Summary of Changes

### 1. **Dataset Loading** (`dataset_loader.py`)
**NEW FILE** - Unified data management system

- **SamsungHRVDatasetLoader**: Loads real Samsung HRV + Sleep Diary dataset
  - Aligns HRV metrics (SDNN, SDSD, LF, HF, RMSSD) with sleep diary entries
  - Maps GAD-7 scores to severity labels (minimal/mild/moderate/severe)
  - Computes rolling 7-day and 14-day aggregate features per user
  
- **DatasetManager**: Unified interface with fallback support
  - Tries to load real dataset first
  - Falls back to synthesized data if real dataset unavailable
  - Ensures backward compatibility

- **Feature Alignment**: Handles missing features gracefully
  - Fills missing columns with means
  - Maintains schema consistency across data sources

**Usage:**
```python
from dataset_loader import DatasetManager, get_feature_aligned_dataframe
manager = DatasetManager(use_real_data=True)
df, metadata = manager.load_data()
```

---

### 2. **Model Evaluation** (`evaluate_model.py`)
**NEW FILE** - Healthcare ML best practices

- **Leave-One-Subject-Out (LOSO) Cross-Validation**
  - Critical for healthcare: trains on N-1 subjects, tests on 1
  - Prevents information leakage from same subject
  - Simulates real clinical deployment

- **Stratified k-Fold Cross-Validation**
  - Standard 5-fold stratified splitting
  - Preserves class distribution across folds
  - Comparison baseline for LOSO results

- **Comprehensive Metrics**
  - Accuracy, Macro F1, Weighted F1, AUC-ROC, Cohen's Kappa
  - Per-class precision/recall/F1
  - Confusion matrices for all folds
  - Formatted summary tables

- **Model Comparison**
  - Side-by-side evaluation: XGBoost vs LSTM vs Ensemble
  - JSON export for reproducibility
  - HTML summary table generation

**Usage:**
```python
from evaluate_model import evaluate_xgboost
loso_results, stratified_results, evaluator = evaluate_xgboost(df)
evaluator.print_formatted_summary(results)
```

---

### 3. **Temporal LSTM Model** (`temporal_model.py`)
**NEW FILE** - Sequence modeling with PyTorch

- **LSTMModel Architecture**
  ```
  Input: (batch, 14 days, n_features)
  ├─ LSTM(64, bidirectional) + Dropout(0.2)
  ├─ LSTM(32, bidirectional) + Dropout(0.2)
  ├─ Dense(16, ReLU)
  └─ Dense(4, softmax) → [Minimal, Mild, Moderate, Severe]
  ```

- **TemporalDataset**: Builds sliding windows from time-series data
  - Lookback window: 14 days by default
  - Per-user sequence construction
  - DataLoader compatibility

- **LSTMTrainer**: Full training pipeline
  - Adam optimizer with weight decay
  - Early stopping on validation loss (patience=10)
  - Training history tracking
  - Best model checkpointing

- **ModelEnsemble**: Combines XGBoost + LSTM
  - Weighted averaging of probabilities
  - Configurable model weights
  - Graceful fallback if models unavailable

- **Model Persistence**
  - Saves to: `mental_twin_lstm_model.pt`
  - Config saved to: `mental_twin_lstm_config.json`

**Usage:**
```python
from temporal_model import LSTMModel, LSTMTrainer, TemporalDataset

model = LSTMModel(n_features=34)
trainer = LSTMTrainer(model)
trainer.fit(train_loader, val_loader, epochs=100, patience=10)

from temporal_model import save_lstm_model
save_lstm_model(trainer)
```

**Fallback Strategy**: If PyTorch is unavailable, system gracefully falls back to XGBoost-only mode.

---

### 4. **Feature Drift Monitoring** (`drift_monitor.py`)
**NEW FILE** - Population-level feature drift detection

- **DriftMonitor Class**: Comprehensive drift detection
  - Baseline: Feature distributions from training data
  - Current: Last 7 days of realtime data per user
  - Drift threshold: 30% of features drifted triggers alert

- **Two Detection Methods**
  1. **Evidently Library** (primary): DataDriftPreset report
  2. **Kolmogorov-Smirnov Test** (fallback): Statistical drift test

- **Drift Reports** (HTML format)
  - Feature statistics comparison table
  - Per-feature drift status
  - Visual alerts for data anomalies
  - Saved to: `reports/drift_{user_id}_{timestamp}.html`

- **API Integration**
  - Triggered via `/api/drift_report/{user_id}`
  - Returns JSON status summary
  - HTML report generation

**Usage:**
```python
from drift_monitor import DriftMonitor

baseline_df = pd.read_csv('training_data.csv')
monitor = DriftMonitor(baseline_df=baseline_df)

# Quick status check
status = monitor.get_drift_status(user_id='user1', days=7)

# Full HTML report
report_path = monitor.generate_report(user_id='user1')
```

---

### 5. **SHAP Explainability** (Modified `digital_twin_predictor.py`)

- **SHAPExplainer Class**: Model-agnostic explanations
  - TreeExplainer for XGBoost (main model)
  - DeepExplainer for LSTM (alternative)
  - Per-prediction feature importance

- **Explanation Format**
  ```python
  {
    "feature": "hrv_sdnn_7d_mean",
    "impact": -0.18,
    "direction": "increases_risk",  # or "reduces_risk"
    "value": 42.3
  }
  ```

- **Human-Readable Summaries**
  - Top 5 risk drivers identified
  - Protective factors highlighted
  - Clinical interpretation provided

- **predict_with_explanation()** Function
  - Makes prediction + SHAP explanation in one call
  - Gracefully handles missing SHAP library
  - Returns structured result dict

**Usage:**
```python
from digital_twin_predictor import predict_with_explanation

result = predict_with_explanation(data, user_id='user1', use_shap=True)
print(result['explanation_summary'])
# Output: "Low 7-day HRV and irregular sleep are primary risk drivers..."
```

---

### 6. **Advanced Feature Engineering** (Modified `config.py` + `data_simulation_and_preprocessing.py`)

**New Features Added:**
| Feature | Computation | Clinical Relevance |
|---------|-------------|-------------------|
| `lf_power` | Low frequency HRV component | Sympathetic nervous system activity |
| `hf_power` | High frequency HRV component | Parasympathetic (vagal) tone |
| `lf_hf_ratio` | LF/HF ratio | Sympathovagal balance; stress indicator |
| `rmssd` | Root mean square of successive differences | Heart rate variability measure |
| `sdnn` / `sdsd` | NN interval statistics | Temporal HRV metrics |
| `sleep_onset_consistency_14d` | Std dev of sleep onset times | Circadian rhythm regularity |
| `sleep_debt_14d` | Cumulative sleep deficit vs personal avg | Sleep homeostasis metric |
| `circadian_regularity` | Daily routine consistency score | Lifestyle stability indicator |
| `sentiment_slope_3d` | 3-day rolling sentiment trend | Mood trajectory; therapeutic progress |
| `activity_consistency_7d` | Coefficient of variation (steps) | Physical activity stability |

**Backward Compatibility**: All new features gracefully degrade if unavailable
- Missing features filled with column means
- Old records without new features still usable
- Optional features list maintained in config

**Usage in Pipeline:**
```python
from data_simulation_and_preprocessing import compute_advanced_features

df = compute_advanced_features(df)  # Adds all 10 new features
```

---

### 7. **FastAPI Migration** (`main.py` - replaces Flask `app.py`)

**Production-Ready Infrastructure:**
- ✅ Async/await for all endpoints
- ✅ CORS middleware configured
- ✅ Pydantic models for request/response validation
- ✅ Type hints throughout
- ✅ Automatic API documentation (Swagger UI)
- ✅ Health check endpoint

**Endpoints Preserved** (same paths as Flask):
```
POST   /api/simulate                  → POST request
GET    /api/predict/{user_id}         → Real-time prediction
GET    /api/history/{user_id}         → Historical chart data
GET    /api/report/{user_id}          → Clinical report download
GET    /api/user_profiles             → Available users dropdown
POST   /api/ingest_manual/{user_id}   → Manual data ingestion
GET    /api/latest_data/{user_id}     → Last N data points
```

**New Endpoints:**
```
GET    /api/drift_report/{user_id}    → Feature drift detection
GET    /api/explain/{user_id}         → SHAP explanations
GET    /health                        → System health check
```

**Request/Response Models** (Pydantic):
```python
class PredictionResponse(BaseModel):
    status: str
    severity: Optional[str]
    confidence: Optional[float]
    gad7: Optional[int]
    timestamp: str
    shap_explanations: Optional[List[Dict]]
    explanation_summary: Optional[str]
```

**Run FastAPI:**
```bash
# Development with auto-reload
uvicorn main:app --reload --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

### 8. **Enhanced Clinical Reports** (`clinical_report_generator.py`)
**NEW FILE** - Professional clinical documentation

- **Comprehensive Report Contents**
  - 30-day longitudinal trend chart (matplotlib → embedded PNG)
  - GAD-7 score tracking with severity bands
  - Estimated PHQ-9 scores (derived from GAD-7)
  - Top 5 SHAP feature drivers as summary table
  - Clinical recommendations based on severity
  - Full clinical disclaimer section

- **Multiple Output Formats**
  - **Markdown** (.md): Human-readable, version-control friendly
  - **HTML**: Web display with embedded images
  - **PDF**: Professional print format (if reportlab installed)

- **Visualizations**
  - 30-day trend chart with severity bands
  - GAD-7 and estimated PHQ-9 overlaid
  - Color-coded severity regions (green/yellow/orange/red)
  - Interactive date labels

- **Clinical Mappings**
  - GAD-7 to PHQ-9 score correlation (~0.85 + 2)
  - Severity interpretation (minimal/mild/moderate/severe)
  - Clinical recommendations per severity level
  - HIPAA-compliant privacy statement

**Usage:**
```python
from clinical_report_generator import generate_comprehensive_report

reports = generate_comprehensive_report(
    user_id='user1',
    history_df=history_df,
    shap_explanations=shap_results
)

# Outputs: reports['markdown'], reports['html'], reports['pdf']
```

---

### 9. **Configuration** (Updated `config.py`)

**New Configuration Keys:**
```python
# Dataset configuration
SAMSUNG_DATASET_PATH = None  # Set to path for real data

# Temporal model settings
LSTM_CONFIG = {
    'lookback_days': 14,
    'hidden_size_1': 64,
    'hidden_size_2': 32,
    'dropout': 0.2,
    'n_classes': 4,
    'bidirectional': True,
}

# Evaluation settings
EVALUATION_CONFIG = {
    'loso_enabled': True,
    'stratified_kfold_splits': 5,
    'test_size': 0.1,
}
```

**Feature Lists:**
- `CORE_FEATURE_BASES`: Core physiological features
- `NON_LAGGED_SUMMARY_BASES`: Non-temporal summary features
- `ADVANCED_FEATURES`: New 10 engineered features
- `OPTIONAL_FEATURES`: Features that may not always be available
- `FINAL_MODEL_FEATURES`: Complete feature list (all features combined)

---

### 10. **Dependencies** (Updated `requirements.txt`)

**New packages added:**
```
fastapi>=0.95.0          # Modern async web framework
uvicorn>=0.21.0          # ASGI server
pydantic>=1.9.0          # Request validation
shap>=0.41.0             # Model explainability
evidently>=0.2.8         # Drift detection
matplotlib>=3.5.0        # Visualization
seaborn>=0.11.0          # Statistical plots
reportlab>=3.6.0         # PDF generation (optional)
weasyprint>=57.0         # PDF generation alternative
python-multipart>=0.0.5  # FastAPI file upload support
```

---

## 🚀 Quick Start Guide

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Generate Synthetic Data** (or use real Samsung dataset)
```bash
python data_simulation_and_preprocessing.py
```

### 3. **Train XGBoost Model**
```bash
python digital_twin_model.py
```

### 4. **Evaluate with Rigorous CV**
```bash
python evaluate_model.py
```

### 5. **Train LSTM (Optional)**
```python
from dataset_loader import DatasetManager
from temporal_model import TemporalDataset, LSTMModel, LSTMTrainer, save_lstm_model

manager = DatasetManager()
df, _ = manager.load_data()

dataset = TemporalDataset(df)
X, y = dataset.build_sequences()

model = LSTMModel(n_features=X.shape[2])
trainer = LSTMTrainer(model)
# ... train with dataloaders ...
save_lstm_model(trainer)
```

### 6. **Start FastAPI Server**
```bash
uvicorn main:app --reload --port 8000
```

### 7. **Access API**
- Dashboard: http://localhost:8000/
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

## 📊 API Examples

### Get Prediction with SHAP Explanation
```bash
curl http://localhost:8000/api/predict/user1_data_baseline
```

Response:
```json
{
  "status": "success",
  "severity": "mild",
  "confidence": 0.73,
  "gad7": 7,
  "timestamp": "2026-05-12T10:30:45.123456",
  "shap_explanations": [
    {
      "feature": "hrv_sdnn_7d_mean",
      "impact": -0.18,
      "direction": "increases_risk",
      "value": 42.3
    },
    ...
  ],
  "explanation_summary": "Low 7-day HRV and irregular sleep are primary risk drivers..."
}
```

### Get Drift Report
```bash
curl http://localhost:8000/api/drift_report/user1_data_baseline?days=7
```

Response:
```json
{
  "status": "normal",
  "timestamp": "2026-05-12T10:30:45",
  "drifted_features_count": 2,
  "drift_percentage": 0.08,
  "summary": "No significant drift detected (8% features drifted)",
  "report_path": "reports/drift_user1_20260512_103045.html"
}
```

### Get Clinical Report
```bash
curl http://localhost:8000/api/report/user1_data_baseline \
  -H "Accept: text/markdown" > report.md
```

---

## 🔧 Configuration Guide

### Using Real Samsung Dataset
```python
# In config.py
SAMSUNG_DATASET_PATH = '/path/to/samsung_dataset'

# Then in code:
from dataset_loader import DatasetManager
manager = DatasetManager(use_real_data=True)
df, metadata = manager.load_data()
```

### Customizing LSTM Architecture
```python
# In config.py
LSTM_CONFIG = {
    'lookback_days': 21,  # 3 weeks instead of 2
    'hidden_size_1': 128,  # Larger first layer
    'hidden_size_2': 64,   # Larger second layer
    'dropout': 0.3,        # More regularization
    'bidirectional': True,
}
```

### Changing Drift Threshold
```python
from drift_monitor import DriftMonitor

monitor = DriftMonitor(baseline_df=df, drift_threshold=0.25)  # Alert if 25%+ features drift
```

---

## 📈 Evaluation Results Format

The `evaluation_results.json` contains:
```json
{
  "xgboost_loso": {
    "accuracy": 0.8421,
    "macro_f1": 0.7834,
    "weighted_f1": 0.8312,
    "cohens_kappa": 0.7654,
    "auc_roc": 0.9123,
    "cv_type": "LOSO",
    "num_folds": 4,
    "confusion_matrix": [[...], [...], [...], [...]],
    "class_report": {...}
  },
  "xgboost_stratified": {
    ...
  }
}
```

---

## 🛡️ Backward Compatibility

✅ **All existing code remains functional:**
- Original `app.py` (Flask) still works; `main.py` is new alternative
- Synthesized data generation unchanged
- XGBoost training pipeline preserved
- SQLite database format unchanged
- `digital_twin_predictor.py` enhanced, not replaced

✅ **Graceful degradation:**
- SHAP explanations optional (library check)
- Temporal LSTM optional (PyTorch check)
- Drift monitoring optional (evidently check)
- PDF generation optional (reportlab check)
- Real dataset optional (falls back to synthesis)

---

## 🔐 Security & Privacy

- **CORS Configured**: Only allows same-origin requests by default (configure as needed)
- **Pydantic Validation**: All API inputs validated
- **Error Handling**: No sensitive information in error messages
- **Audit Ready**: Structured logging for compliance

---

## 📝 Testing Checklist

- [ ] Run `evaluate_model.py` to verify LOSO CV works
- [ ] Generate synthetic data with new features
- [ ] Start FastAPI server and access `/docs`
- [ ] Call `/api/predict/{user_id}` and verify SHAP explanations
- [ ] Call `/api/drift_report/{user_id}` and check HTML output
- [ ] Generate clinical report and verify visualizations
- [ ] Test both XGBoost and LSTM predictions (if available)

---

## 📚 References

- **SHAP**: https://shap.readthedocs.io/
- **evidently**: https://docs.evidentlyai.com/
- **FastAPI**: https://fastapi.tiangolo.com/
- **PyTorch LSTM**: https://pytorch.org/docs/stable/nn.html#lstm
- **XGBoost**: https://xgboost.readthedocs.io/

---

## 🤝 Support

For issues or questions about the implementation:
1. Check the docstrings in each module
2. Review the type hints for expected inputs/outputs
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
4. Verify dependencies with: `pip list`

---

**Last Updated:** May 12, 2026  
**Version:** 2.0.0
