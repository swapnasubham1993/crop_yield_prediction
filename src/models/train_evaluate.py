import pandas as pd
import numpy as np
import yaml
import os
import pickle
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import mlflow
import mlflow.sklearn

# Linear
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# Tree
from sklearn.tree import DecisionTreeRegressor
# Ensemble
from sklearn.ensemble import (
    RandomForestRegressor, 
    AdaBoostRegressor, 
    GradientBoostingRegressor,
    BaggingRegressor,
    VotingRegressor,
    StackingRegressor
)
# Neighbors
from sklearn.neighbors import KNeighborsRegressor

# Boosting (External)
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def evaluate_models():
    params = load_params()
    
    # Paths
    train_path = params['data']['train_file']
    test_path = params['data']['test_file']
    target_col = params['data']['target_col'].lower()
    
    # Load Data
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Define Models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Bagging": BaggingRegressor()
    }
    
    # Add external boosting libraries if available
    if XGBRegressor:
        models["XGBoost"] = XGBRegressor()
    else:
        print("XGBoost not installed. Skipping.")
        
    if LGBMRegressor:
        models["LightGBM"] = LGBMRegressor(verbose=-1)
    else:
        print("LightGBM not installed. Skipping.")
        
    if CatBoostRegressor:
        models["CatBoost"] = CatBoostRegressor(verbose=0)
    else:
        print("CatBoost not installed. Skipping.")
        
    # Voting Regressor (using a subset of base models)
    voting_estimators = [
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=50)),
        ('gb', GradientBoostingRegressor(n_estimators=50))
    ]
    models["Voting Regressor"] = VotingRegressor(estimators=voting_estimators)
    
    # Stacking Regressor
    stacking_estimators = [
        ('vr', Ridge()),
        ('rf', RandomForestRegressor(n_estimators=50))
    ]
    models["Stacking Regressor"] = StackingRegressor(
        estimators=stacking_estimators,
        final_estimator=LinearRegression()
    )
    
    # Evaluation
    results = []
    
    print("\nStarting Model Evaluation...")
    print("-" * 80)
    print(f"{'Model':<25} | {'MAE':<10} | {'MSE':<10} | {'R2 Score':<10} | {'Time (s)':<10}")
    print("-" * 80)
    
    best_model_name = None
    best_r2 = -float("inf")
    best_model_obj = None
    
    mlflow.set_experiment(params['base']['project'])
    
    with mlflow.start_run(run_name="Model Comparison"):
        for name, model in models.items():
            run_name = name.replace(" ", "_")
            with mlflow.start_run(run_name=run_name, nested=True):
                start_time = time.time()
                
                try:
                    # Train
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    elapsed = time.time() - start_time
                    
                    # Log to MLflow
                    mlflow.log_metric("MAE", mae)
                    mlflow.log_metric("MSE", mse)
                    mlflow.log_metric("R2", r2)
                    mlflow.log_metric("Training_Time", elapsed)
                    mlflow.log_param("model_type", name)
                    
                    # Log Model (optional, can be heavy for many models)
                    # mlflow.sklearn.log_model(model, "model") 
                    
                    print(f"{name:<25} | {mae:<10.4f} | {mse:<10.4f} | {r2:<10.4f} | {elapsed:<10.2f}")
                    
                    results.append({
                        "Model": name,
                        "MAE": mae,
                        "MSE": mse,
                        "R2": r2,
                        "Time": elapsed
                    })
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_name = name
                        best_model_obj = model
                        
                except Exception as e:
                    print(f"{name:<25} | ERROR: {str(e)}")
            
    print("-" * 80)
    print(f"Best Model: {best_model_name} (R2: {best_r2:.4f})")
    
    # Save Results
    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/model_comparison.csv", index=False)
    print("Comparison results saved to reports/model_comparison.csv")
    
    # Save Best Model
    if best_model_obj:
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model_obj, f)
        print(f"Best model ({best_model_name}) saved to {model_path}")

if __name__ == "__main__":
    evaluate_models()
