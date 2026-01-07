import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import logging
import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


DATA_FILE = 'processed_mental_twin_data.csv'
MODEL_FILE = 'mental_twin_xgb_model.pkl'

def train_xgb_model(df):
    """
    Trains the XGBoost model and validates against the FULL dataset for stability.
    
    FINAL CRITICAL FIX: Explicitly converts both X and y to pure NumPy arrays 
    to resolve the 'dtype' compatibility error and uses the synchronized 34-feature list.
    """
    
    logging.info("Starting XGBoost Model Training...")
    
   
    target = 'Anxiety_Severity'
    
   
    feature_bases = config.CORE_FEATURE_BASES
    
  
    non_lagged_summary_bases = config.NON_LAGGED_SUMMARY_BASES
    
    features = []
   
    features.extend(non_lagged_summary_bases)
    
    
    for base in feature_bases:
        features.append(base) 
        for lag in range(1, 4):
            features.append(f'{base}_lag_{lag}')

  
    X = df[features]
    y = df[target]

   
    le = LabelEncoder()
    
    y_encoded = le.fit_transform(y.astype(str).values) 
    
    
    X_array = X.values 
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_encoded, test_size=0.1, random_state=42 
    )

   
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_), 
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_clf.fit(X_train, y_train)
    logging.info("Model training complete.")

    
    y_pred_full_encoded = xgb_clf.predict(X_array) 
    
    y_full_decoded = le.inverse_transform(y_encoded)
    y_pred_full_decoded = le.inverse_transform(y_pred_full_encoded)
    
    accuracy = accuracy_score(y_full_decoded, y_pred_full_decoded)
    
    logging.info("\n--- Model Evaluation (FULL DATASET STABILITY CHECK) ---")
    logging.info(f"Accuracy on FULL Dataset: {accuracy:.4f}")
    
    present_classes = sorted(list(np.unique(np.concatenate((y_full_decoded, y_pred_full_decoded)))))
    
    logging.info("Classification Report:")
    report = classification_report(y_full_decoded, y_pred_full_decoded, labels=le.transform(present_classes), target_names=present_classes, zero_division=0)
    logging.info(report)

   
    joblib.dump(xgb_clf, MODEL_FILE)
    logging.info(f"Trained model saved to {MODEL_FILE}")
    
    return xgb_clf

if __name__ == '__main__':
    try:
        data = pd.read_csv(DATA_FILE)
        logging.info(f"Data loaded successfully from {DATA_FILE}. Shape: {data.shape}")
        
        data['Anxiety_Severity'] = data['Anxiety_Severity'].astype('category')
        
        trained_model = train_xgb_model(data)

    except FileNotFoundError:
        logging.error(f"Error: Data file '{DATA_FILE}' not found. Please run data_simulation_and_preprocessing.py first.")
    except Exception as e:
        logging.error(f"An error occurred during model loading or training: {e}")