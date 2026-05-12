# Mental Health Digital Twin - Project Report

## Overview
The Mental Health Digital Twin is a comprehensive application designed to monitor, predict, and analyze a user's mental health state using physiological and behavioral data. The system utilizes machine learning (XGBoost/LSTM) to predict anxiety severity (measured by the GAD-7 scale) based on daily health metrics.

This project is built using **FastAPI** for the backend, serving a dashboard and providing multiple RESTful API endpoints for prediction, simulation, report generation, and drift monitoring.

## System Architecture

### 1. Data Sources & Inputs
The system ingests physiological and behavioral data either from real datasets (e.g., Samsung HRV + Sleep Diary dataset) or through manual input/simulation. 

**Core Input Features:**
- **Physiological Metrics:** Heart Rate Variability (HRV), Blood Pressure (Systolic/Diastolic), Sleep Duration.
- **Behavioral Metrics:** Step count, Audio levels (from ambient noise/speech), Mood logs.
- **Derived/Advanced Features:** LF/HF power ratio (sympathovagal balance), Sleep debt, Circadian regularity, and Sentiment slope (derived from mood logs).

### 2. Machine Learning Models
The application employs robust machine learning models to analyze inputs and predict mental health severity:
- **XGBoost Classifier (Primary):** Trains on historical data to predict the current mental health severity level.
- **LSTM Model (Temporal - Optional):** A sequence model built with PyTorch that looks at the past 14 days of data to understand temporal trends in the user's health.
- **Explainable AI (SHAP):** The system uses SHAP (SHapley Additive exPlanations) to explain *why* a prediction was made, highlighting which features (e.g., low HRV, high sleep debt) increased or decreased the risk.

### 3. Backend & Endpoints
The backend is powered by FastAPI, offering the following capabilities:

| Endpoint | Method | Input | Output / What Happens |
| :--- | :--- | :--- | :--- |
| `/api/simulate` | `POST` | Trigger via UI | Simulates new data points for a baseline user and writes them to the SQLite database. |
| `/api/ingest_manual/{user_id}` | `POST` | Manual data payload (HRV, steps, sleep, etc.) | Ingests a single user-provided data point into the system for real-time monitoring. |
| `/api/predict/{user_id}` | `GET` | `user_id` (Path) | Fetches the user's latest data, runs the XGBoost/LSTM model, and returns the predicted severity (Minimal, Mild, Moderate, Severe), confidence score, and SHAP explanations. |
| `/api/history/{user_id}` | `GET` | `user_id` (Path) | Generates and returns a 30-day prediction history including dates, GAD-7 scores, and predicted severities for charting. |
| `/api/report/{user_id}` | `GET` | `user_id` (Path) | Generates a comprehensive clinical Markdown report containing longitudinal trends, average GAD-7, and clinical disclaimers. |
| `/api/explain/{user_id}` | `GET` | `user_id` (Path) | Retrieves the detailed SHAP explanation for the latest prediction, returning a human-readable summary of risk drivers. |
| `/api/drift_report/{user_id}` | `GET` | `user_id` (Path), `days` (Query) | Compares the last `N` days of real-time data against the training baseline to detect data drift, returning a drift percentage and generating an HTML report. |

### 4. Data Flow: What Happens with an Input?
When new data is ingested (either via simulation or manual input):
1. **Ingestion & Storage:** The data point (e.g., a day's worth of HRV, sleep, and steps) is received and saved to a local SQLite database (`realtime_data.db`).
2. **Preprocessing:** The `dataset_loader` and preprocessing modules align the new data with expected features, calculating rolling averages (e.g., 7-day HRV mean) and imputing any missing values.
3. **Prediction:** When the `/api/predict` endpoint is called, the preprocessed data is fed into the XGBoost model. The model calculates the probability of each severity class.
4. **Explanation:** SHAP values are computed simultaneously to determine which specific inputs heavily influenced the prediction.
5. **Output:** The frontend dashboard receives the severity label (e.g., "Mild"), the estimated GAD-7 score, and a breakdown of contributing factors (e.g., "Irregular sleep increased risk").

## Key Features for Mentor Review
- **Clinical Rigor:** Models are evaluated using Leave-One-Subject-Out (LOSO) Cross-Validation to prevent data leakage and simulate real clinical deployment.
- **Feature Drift Monitoring:** Incorporates production-level monitoring to detect if incoming user data distributions drift away from the training data over time.
- **Explainability:** Transitions from a "black box" model to a transparent one, making it safer and more useful for potential clinical reviews.
- **Robust Fallbacks:** The system gracefully degrades if advanced modules (like PyTorch for LSTM or SHAP) are unavailable, falling back to basic XGBoost predictions.
