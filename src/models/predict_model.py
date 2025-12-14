import pandas as pd
import pickle
import os
import yaml
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.logger import get_logger

logger = get_logger(__name__)

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def predict():
    params = load_params()
    
    # Paths
    model_path = "models/best_model.pkl"
    test_path = params['data']['test_file']
    target_col = params['data']['target_col'].lower()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}. Run training first.")
        
    # Load Model
    logger.info(f"Loading best model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    logger.info(f"Model type: {type(model).__name__}")
        
    # Load Test Data (Simulating new data)
    logger.info(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path)
    
    X_test = df_test.drop(columns=[target_col])
    y_true = df_test[target_col]
    
    # Predict
    logger.info("Running predictions...")
    predictions = model.predict(X_test)
    
    # Show samples
    results = pd.DataFrame({
        "Actual": y_true,
        "Predicted": predictions,
        "Difference": y_true - predictions
    })
    
    logger.info(f"\nSample Predictions:\n{results.head(10).to_string()}")
    
    # Save predictions
    save_path = "reports/predictions.csv"
    results.to_csv(save_path, index=False)
    logger.info(f"\nAll predictions saved to {save_path}")

if __name__ == "__main__":
    predict()
