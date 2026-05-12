# Mental Health Digital Twin

An anxiety-severity digital twin that combines physiological signals, sentiment analysis, temporal modeling, explainability, and drift monitoring. The system predicts GAD-7 severity classes and produces clinical-style summaries for monitoring purposes.

## What’s Included

- Real dataset support via [dataset_loader.py](dataset_loader.py) for the Samsung HRV + Sleep Diary dataset, with a fallback to synthesized data.
- Rigorous evaluation via [evaluate_model.py](evaluate_model.py) using Leave-One-Subject-Out and stratified 5-fold cross-validation.
- Temporal modeling via [temporal_model.py](temporal_model.py) with a PyTorch LSTM and XGBoost ensemble support.
- Explainable predictions via SHAP in [digital_twin_predictor.py](digital_twin_predictor.py).
- Drift monitoring via [drift_monitor.py](drift_monitor.py).
- FastAPI backend in [main.py](main.py), serving the existing dashboard in [index.html](index.html).
- Enhanced clinical reports in [clinical_report_generator.py](clinical_report_generator.py).

## Architecture

- [data_simulation_and_preprocessing.py](data_simulation_and_preprocessing.py) generates synthetic training data and computes additional engineered features.
- [digital_twin_model.py](digital_twin_model.py) trains the XGBoost model and saves `mental_twin_xgb_model.pkl`.
- [digital_twin_predictor.py](digital_twin_predictor.py) loads realtime rows from SQLite, assembles the feature vector, predicts severity, and returns SHAP explanations.
- [sqlite_data_writer.py](sqlite_data_writer.py) remains the default realtime storage layer.
- [influxdb_data_writer.py](influxdb_data_writer.py) stays optional.
- [config.py](config.py) defines feature schemas, user profiles, and model settings.

## Key Outputs

- `processed_mental_twin_data.csv` - engineered training dataset.
- `mental_twin_xgb_model.pkl` - trained XGBoost model.
- `mental_twin_lstm_model.pt` - optional temporal model checkpoint.
- `realtime_data.db` - SQLite realtime store.
- `evaluation_results.json` - cross-validation results.
- `reports/` - generated drift and clinical reports.

## Runtime Features

- GAD-7 severity classes: minimal, mild, moderate, severe.
- SHAP top feature contributions for each prediction.
- 30-day longitudinal clinical report with embedded chart and disclaimer.
- Weekly drift checks against recent realtime data.
- Backward compatibility for older records that do not contain new engineered features.

## Quick Start

```bash
cd /path/to/mental-health_digital_twin
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python data_simulation_and_preprocessing.py
python digital_twin_model.py
python evaluate_model.py
uvicorn main:app --reload --port 8000
```

## Main API Endpoints

- `POST /api/simulate`
- `GET /api/predict/{user_id}`
- `GET /api/history/{user_id}`
- `GET /api/report/{user_id}`
- `GET /api/user_profiles`
- `GET /api/explain/{user_id}`
- `GET /api/drift_report/{user_id}`

## Notes

- Python 3.8+ is supported.
- PyTorch, SHAP, evidently, and PDF generation are optional; the system degrades gracefully if those packages are missing.
- SQLite remains the default storage path, and InfluxDB stays optional.
- The dashboard frontend is unchanged.

## Clinical Disclaimer

This tool is not a diagnostic instrument. Predictions are for monitoring purposes only and should be reviewed by a qualified mental health professional.
