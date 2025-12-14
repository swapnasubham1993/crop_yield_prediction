import pandas as pd
import yaml
import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
# Add other imports if needed for dynamic model loading

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def get_model(model_type, params):
    # Simple factory
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=params['train']['n_estimators'],
            learning_rate=params['train']['learning_rate'],
            max_depth=params['train']['max_depth'],
            random_state=params['base']['random_state']
        )
    elif model_type == "random_forest":
         return RandomForestRegressor(
            n_estimators=params['train']['n_estimators'],
            random_state=params['base']['random_state']
        )
    # Default fallback
    return GradientBoostingRegressor()

def train():
    params = load_params()
    
    train_path = params['data']['train_file']
    target_col = params['data']['target_col'].lower() # Ensure lower case match
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load Data
    print(f"Loading training data from {train_path}...")
    df = pd.read_csv(train_path)
    
    X_train = df.drop(columns=[target_col])
    y_train = df[target_col]
    
    # Init Model
    model_type = params['train']['model_type']
    model = get_model(model_type, params)
    
    # MLflow Tracking
    mlflow.set_experiment(params['base']['project'])
    
    with mlflow.start_run():
        print(f"Training {model_type}...")
        model.fit(X_train, y_train)
        
        # Log Params
        mlflow.log_params(params['train'])
        mlflow.log_param("model_type", model_type)
        
        # Save Model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        print(f"Model saved to {model_path}")
        
        # Log Artifact
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train()
