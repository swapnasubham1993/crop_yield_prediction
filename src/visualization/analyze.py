import pandas as pd
import yaml
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def analyze():
    params = load_params()
    # Read from INTERIM (cleaned) data now
    clean_path = "data/interim/cleaned_data.csv"
    report_dir = "reports/figures"
    stats_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"Loading data for analysis from {clean_path}...")
    df = pd.read_csv(clean_path)
    
    # --- Textual Statistics ---
    buffer = io.StringIO()
    
    buffer.write("=== DATA SHAPE ===\n")
    buffer.write(f"{df.shape}\n\n")
    
    buffer.write("=== COLUMNS ===\n")
    buffer.write(f"{list(df.columns)}\n\n")
    
    buffer.write("=== INFO ===\n")
    df.info(buf=buffer)
    buffer.write("\n\n")
    
    buffer.write("=== NUMERIC DESCRIPTIVE STATS ===\n")
    buffer.write(f"{df.describe(include=['int64', 'float64']).T.to_string()}\n\n")
    
    buffer.write("=== CATEGORICAL DESCRIPTIVE STATS ===\n")
    try:
        buffer.write(f"{df.describe(include=['object']).T.to_string()}\n\n")
    except ValueError:
        buffer.write("No object columns found.\n\n")
        
    buffer.write("=== MISSING VALUES ===\n")
    buffer.write(f"{df.isnull().sum()}\n\n")
    
    buffer.write("=== DUPLICATE VALUES ===\n")
    buffer.write(f"Count: {df.duplicated().sum()}\n\n")
    
    buffer.write("=== CROP YEAR COUNTS ===\n")
    if 'crop_year' in df.columns:
        buffer.write(f"{df['crop_year'].value_counts().to_string()}\n\n")

    buffer.write("=== CROP COUNTS ===\n")
    if 'crop' in df.columns:
        buffer.write(f"{df['crop'].value_counts().to_string()}\n\n")

    buffer.write("=== SEASON COUNTS ===\n")
    if 'season' in df.columns:
        buffer.write(f"{df['season'].value_counts().to_string()}\n\n")

    buffer.write("=== STATE COUNTS ===\n")
    if 'state' in df.columns:
        buffer.write(f"{df['state'].value_counts().to_string()}\n\n")

    buffer.write("=== AVERAGE YIELD PER STATE ===\n")
    if 'state' in df.columns and 'yield' in df.columns:
        yield_per_state = df.groupby('state')['yield'].mean().sort_values(ascending=False)
        buffer.write(f"{yield_per_state.to_string()}\n\n")
    
    # Save Report
    with open(os.path.join(stats_dir, "eda_summary.txt"), "w", encoding='utf-8') as f:
        f.write(buffer.getvalue())
    print(f"EDA Summary saved to {os.path.join(stats_dir, 'eda_summary.txt')}")

    # --- Visual Analysis ---
    
    # 2. Correlation Matrix (Numeric)
    # 2. Correlation Matrix (Numeric)
    plt.figure(figsize=(15, 10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'correlation_matrix.png'))
        plt.close()

    # 3. Text & Visual Analysis per Metric
    metrics = {
        'Yield': 'yield', 
        'Annual_Rainfall': 'annual_rainfall', 
        'Area': 'area', 
        'Production': 'production', 
        'Fertilizer': 'fertilizer', 
        'Pesticide': 'pesticide'
    }

    for metric_name, metric_col in metrics.items():
        if metric_col not in df.columns:
            print(f"Skipping {metric_name} - column '{metric_col}' not found.")
            continue
            
        print(f"Generating plots for {metric_name}...")
        
        # A. Metric by State (Sum)
        if 'state' in df.columns:
            df_state = df.groupby('state', as_index=False)[metric_col].sum()
            plt.figure(figsize=(15, 5))
            sns.barplot(x='state', y=metric_col, data=df_state, palette='gnuplot')
            plt.xticks(rotation=90)
            plt.title(f'{metric_name} by State')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, f'{metric_col}_by_state.png'))
            plt.close()

            # B. Top 10 States by Metric (Sum)
            df_state_sort = df_state.sort_values(by=metric_col, ascending=False)
            top_10_states = df_state_sort.head(10)
            
            plt.figure(figsize=(15, 5))
            sns.barplot(x='state', y=metric_col, data=top_10_states, palette='gnuplot')
            plt.xticks(rotation=90)
            plt.title(f'Top 10 States by {metric_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, f'{metric_col}_top_10_states.png'))
            plt.close()

        # C. Metric over Year (Sum)
        if 'crop_year' in df.columns:
            df_year = df.groupby('crop_year', as_index=False)[metric_col].sum()
            plt.figure(figsize=(15, 5))
            plt.plot(df_year['crop_year'], df_year[metric_col], color='blue', linestyle='dashed', marker='o', markersize=12, markerfacecolor='yellow')
            plt.xlabel('Year')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} over the Years')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, f'{metric_col}_over_years.png'))
            plt.close()
    
    print(f"Analysis complete. Figures saved to {report_dir}")

if __name__ == "__main__":
    analyze()
