# Mental Health Digital Twin - Setup & Run Guide

This guide provides step-by-step instructions for setting up and running the Mental Health Digital Twin project on a new system from scratch.

## Prerequisites
- **Python 3.9+** installed on your system.
- Git (if cloning the repository).

## 1. Setup Python Environment
It is highly recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

## 2. Generate Data & Databases
Since the project relies on a processed dataset and a real-time SQLite database, you need to generate these first. The simulation script will create synthetic baseline data, calculate all advanced features, and initialize the databases.

```bash
# Generate synthetic baseline data and initialize SQLite database
python data_simulation_and_preprocessing.py
```
*Expected Output:* This will create `processed_mental_twin_data.csv` (used for training) and `realtime_data.db` (used for real-time user ingestion).

## 3. Train the Core Model
Once the data is generated, train the XGBoost machine learning model. This model predicts the anxiety severity (GAD-7) based on the physiological and behavioral metrics.

```bash
# Train the XGBoost model
python digital_twin_model.py
```
*Expected Output:* This will create the `mental_twin_xgb_model.pkl` file, which is the trained model file used by the API.

## 4. Run the Application Server
With the database initialized and the model trained, you can now start the FastAPI server. 

```bash
# Start the FastAPI server (with auto-reload enabled)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 5. Access the Application
Once the server is running, you can access the application through your web browser:

- **Main Dashboard (UI):** [http://localhost:8000/](http://localhost:8000/)
- **Interactive API Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative API Documentation (ReDoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Troubleshooting
- **Missing Modules Error:** If the server starts but warns about missing modules (like SHAP or PyTorch), ensure all dependencies in `requirements.txt` are installed. The system has graceful fallbacks, but full functionality requires all packages.
- **Data Not Found:** If predictions fail, make sure you successfully ran Step 2 and Step 3, and that the `.csv`, `.db`, and `.pkl` files exist in the project root directory.
