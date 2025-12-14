import pandas as pd
import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.logger import get_logger

logger = get_logger(__name__)

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def ingest_data():
    params = load_params()
    
    # Paths
    raw_path = params['data']['raw_file']
    interim_dir = "data/interim"
    os.makedirs(interim_dir, exist_ok=True)
    clean_path = os.path.join(interim_dir, "cleaned_data.csv")
    
    # Load
    logger.info(f"Loading data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Standardize Column Names
    # Lowercase, strip, replace space/special chars with underscore
    df.columns = df.columns.str.lower().str.strip().str.replace(r'[^a-zA-Z0-9]', '_', regex=True)
    
    # Standardize String Values
    # Lowercase, strip, replace space/special chars with underscore for all object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.lower().str.strip().str.replace(r'\s+', '_', regex=True)
    
    logger.info(f"Data cleaned (string standardization). saving to {clean_path}")
    df.to_csv(clean_path, index=False)
    
    # MLflow Logging
    import mlflow
    mlflow.set_experiment(params['base']['project'])
    with mlflow.start_run(run_name="Data Ingestion"):
        mlflow.log_param("raw_file", raw_path)
        mlflow.log_metric("raw_rows", df.shape[0])
        mlflow.log_metric("raw_cols", df.shape[1])
        mlflow.log_artifact(clean_path)
        logger.info("Logged data ingestion metrics to MLflow.")

if __name__ == "__main__":
    ingest_data()
