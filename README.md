# Crop Yield Prediction Project

A production-ready machine learning pipeline to predict crop yield based on agricultural and environmental factors.

## 🚀 Quick Start

### 1. Run the Full Model Pipeline
You have two options to run the pipeline:

**Option A: Using DVC (Recommended)**
Data Version Control (DVC) automatically handles dependencies and only reruns what has changed.
```bash
dvc repro
```

**Option B: Manual Python Commands**
Run the scripts in the following order:
```bash
# 1. Ingest Data (Load & Standardize)
python src/data/ingest.py

# 2. Analyze (Generate EDA Report)
python src/visualization/analyze.py

# 3. Preprocess (Feature Engineering, VIF, Scaling)
python src/features/preprocess.py

# 4. Train & Evaluate (Compare 15+ models, Save Best)
python src/models/train_evaluate.py

# 5. Predict (Inference Test)
python src/models/predict_model.py
```

### 2. Launch the Web Application
Start the Streamlit UI to interact with the best model:
```bash
streamlit run app.py
```

---

## 🏗️ Pipeline Details

### Data Ingestion
*   **Script**: `src/data/ingest.py`
*   **Input**: `data/raw/crop_yield.csv`
*   **Output**: `data/interim/cleaned_data.csv`
*   **Details**: Standardizes column names (snake_case) and string values.

### Exploratory Data Analysis (EDA)
*   **Script**: `src/visualization/analyze.py`
*   **Output**: `reports/eda_summary.txt`, `reports/figures/*.png`
*   **Details**: Generates descriptive stats, correlation matrices, and distribution plots.

### Feature Engineering
*   **Script**: `src/features/preprocess.py`
*   **Input**: `data/interim/cleaned_data.csv`
*   **Output**: `data/processed/train.csv`, `test.csv`, `preprocessor.pkl`
*   **Details**:
    *   Drops leakage features (`Area`, `Production`).
    *   Iteratively removes features with high VIF (removed `Fertilizer`).
    *   Applies `PowerTransformer` (Yeo-Johnson).

### Model Training & Evaluation
*   **Script**: `src/models/train_evaluate.py`
*   **Output**: `models/best_model.pkl`, `reports/model_comparison.csv`
*   **Details**: Trains and compares Linear, Tree, implementations (Gradient Boosting, XGBoost, etc.). Saves the best performer automatically.

### Tracking
*   **MLflow**: Metrics, params, and artifacts are logged for Ingest, Preprocess, and Train steps.
*   **DVC**: Data lineage is tracked via `dvc.yaml`.

## 📦 Requirements
*   Python 3.8+
*   `pip install -r requirements.txt`
