import pandas as pd
import yaml
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def preprocess():
    params = load_params()
    
    # Read from INTERIM
    clean_path = "data/interim/cleaned_data.csv"
    processed_dir = params['data']['processed_dir']
    drop_cols = params['data']['drop_cols']
    target_col = params['data']['target_col']
    test_size = params['base']['test_size']
    random_state = params['base']['random_state']
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load
    print(f"Loading cleaned data from {clean_path}...")
    df = pd.read_csv(clean_path)
    
    # Filtering Logic: "count the crop_year and if count is less than 50 then reomves those records"
    if 'crop_year' in df.columns:
        year_counts = df['crop_year'].value_counts()
        years_to_keep = year_counts[year_counts >= 50].index
        initial_shape = df.shape
        df = df[df['crop_year'].isin(years_to_keep)]
        print(f"Filtered years with count < 50. Shape changed from {initial_shape} to {df.shape}")
        
    # Drop Columns
    # Normalize drop_cols to lower case to match cleaned columns
    drop_cols = [c.lower() for c in drop_cols]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # --- Check Skewness ---
    numeric_features = ['annual_rainfall', 'fertilizer', 'pesticide']
    available_numeric = [c for c in numeric_features if c in df.columns]
    
    if available_numeric:
        print("Skewness:\n", df[available_numeric].skew().sort_values())
        
        # Plot Distributions
        plt.figure(figsize=(15, 20))
        for i, col in enumerate(available_numeric):
            plt.subplot(4, 2, i+1)
            sns.histplot(df[col], kde=True, bins=10)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig("reports/figures/feature_distributions.png")
        plt.close()

    # --- Variance Inflation Factor (VIF) ---
    print("Checking VIF...")
    # Select numeric features for VIF (excluding target)
    vif_features = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c != target_col and c not in ['crop_year']] 
    # crop_year is technically numeric but often treated as categorical or dropped. User dropped Area/Production.
    
    # Iterative Removal
    while True:
        if not vif_features:
            break
            
        X_vif = df[vif_features]
        # Add constant for VIF calculation
        X_vif_const = add_constant(X_vif)
        
        vif_data = pd.DataFrame()
        vif_data["feature"] = vif_features
        vif_data["VIF"] = [variance_inflation_factor(X_vif_const.values, i+1) for i in range(len(vif_features))] # +1 to skip const
        vif_data = vif_data.sort_values(by="VIF", ascending=False)
        
        print("\nCurrent VIF:\n", vif_data)
        
        max_vif = vif_data.iloc[0]["VIF"]
        max_feature = vif_data.iloc[0]["feature"]
        
        if max_vif > 10:
            print(f"Removing {max_feature} (VIF={max_vif:.2f} > 10)")
            vif_features.remove(max_feature)
            df = df.drop(columns=[max_feature])
        else:
            print("All features have VIF <= 10. VIF check complete.")
            break
            
    print(f"Final features selected: {list(df.columns)}")
    
    # Separate Target
    target_col = target_col.lower()
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # One-Hot Encoding
    cat_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Convert bool to int
    bool_cols = X_encoded.select_dtypes(include=['bool']).columns
    X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)
    
    # Save Model Columns
    model_columns = list(X_encoded.columns)
    with open(os.path.join(processed_dir, 'columns.pkl'), 'wb') as f:
        pickle.dump(model_columns, f)
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state
    )
    
    print("x_train - >  ", X_train.shape)
    print("y_train - >  ", y_train.shape)
    print("x_test  - >  ", X_test.shape)
    print("y_test  - >  ", y_test.shape)
    
    # Scaling
    pt = PowerTransformer(method='yeo-johnson')
    X_train_scaled = pt.fit_transform(X_train)
    X_test_scaled = pt.transform(X_test)
    
    # Save Preprocessor
    with open(os.path.join(processed_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(pt, f)
        
    # Save Processed Data
    train_df = pd.DataFrame(X_train_scaled, columns=model_columns)
    train_df[target_col] = y_train.values 
    
    test_df = pd.DataFrame(X_test_scaled, columns=model_columns)
    test_df[target_col] = y_test.values
    
    train_df.to_csv(params['data']['train_file'], index=False)
    test_df.to_csv(params['data']['test_file'], index=False)
    
    print(f"Preprocessing complete. Saved to {processed_dir}")

if __name__ == "__main__":
    preprocess()
