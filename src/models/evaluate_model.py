import pandas as pd
import yaml
import os
import pickle
import json
import numpy as np
import mlflow
from sklearn.metrics import r2_score, mean_squared_error

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def evaluate():
    params = load_params()
    
    test_path = params['data']['test_file']
    target_col = params['data']['target_col'].lower()
    model_path = "models/model.pkl"
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # Load Data
    print(f"Loading test data from {test_path}...")
    df = pd.read_csv(test_path)
    
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    
    # Load Model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Predict
    predictions = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    # MLflow (Attach to existing run? Hard to do without ID. Create new 'eval' run or just log via DVC mechanism)
    # We'll just define a DVC metric file.
    
    metrics = {
        "r2_score": r2,
        "rmse": rmse
    }
    
    print(f"Evaluation Metrics: {metrics}")
    
    # Save Metrics for DVC
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Metrics saved to metrics.json")

if __name__ == "__main__":
    evaluate()
