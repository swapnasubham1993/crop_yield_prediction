import streamlit as st
import pandas as pd
import pickle
import os

# Page Config
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌾 Inteligent Crop Yield Prediction")
st.markdown("Enter the agricultural parameters below to predict the crop yield.")

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    # Paths
    model_path = "models/best_model.pkl"
    preprocessor_path = "data/processed/preprocessor.pkl"
    columns_path = "data/processed/columns.pkl"
    data_path = "data/interim/cleaned_data.csv"
    
    if not os.path.exists(model_path):
        st.error("Model not found. Please run training first.")
        return None, None, None, None
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
        
    with open(columns_path, "rb") as f:
        model_columns = pickle.load(f)
        
    # Load data for dropdown options
    df = pd.read_csv(data_path)
    
    return model, preprocessor, model_columns, df

model, preprocessor, model_columns, df = load_artifacts()

if model is not None:
    # --- Sidebar ---
    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses a machine learning model to predict crop yield per unit area "
        "based on environmental and agricultural factors."
    )
    
    # --- Input Form ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location & Season")
        
        # State
        states = sorted(df['state'].unique())
        selected_state = st.selectbox("State", states)
        
        # Season
        seasons = sorted(df['season'].unique())
        selected_season = st.selectbox("Season", seasons)
        
        # Crop
        crops = sorted(df['crop'].unique())
        selected_crop = st.selectbox("Crop", crops)
        
    with col2:
        st.subheader("💧 Environmental Factors")
        
        # Rainfall
        avg_rainfall = df['annual_rainfall'].mean()
        annual_rainfall = st.number_input(
            "Annual Rainfall (mm)", 
            min_value=0.0, 
            value=float(avg_rainfall)
        )
        
        # Pesticide
        avg_pesticide = df['pesticide'].mean()
        pesticide = st.number_input(
            "Pesticide Usage (kg)", 
            min_value=0.0, 
            value=float(avg_pesticide)
        )

    # --- Prediction ---
    if st.button("Predict Yield"):
        # Create DataFrame
        input_data = pd.DataFrame({
            'annual_rainfall': [annual_rainfall],
            'pesticide': [pesticide],
            'crop': [selected_crop],
            'season': [selected_season],
            'state': [selected_state]
        })
        
        # One-Hot Encoding
        # We need to match the columns structure expected by the model
        input_encoded = pd.get_dummies(input_data, columns=['crop', 'season', 'state'], drop_first=True)
        
        # Align with model columns (add missing cols with 0)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Bool to Int conversion (if any from reindex/dummies)
        input_encoded = input_encoded.astype(int)
        
        # Scale (using preprocessor on specific columns if needed, or whole df? 
        # Preprocess.py used PowerTransformer on whole X_train)
        # However, preprocessor was fitted on X_train (encoded).
        # But wait, PowerTransformer usually doesn't like sparse boolean columns with 0/1 unless they are treated as numeric.
        # In preprocess.py: X_train_scaled = pt.fit_transform(X_train)
        # So we must apply it to the whole input_encoded
        
        try:
            input_scaled = preprocessor.transform(input_encoded)
            
            # Predict
            prediction = model.predict(input_scaled)
            
            # Display
            st.success(f"🌱 Predicted Yield: **{prediction[0]:.2f}**")
            
            # Contextual info?
            st.info(f"Using model: {type(model).__name__}")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Please ensure all model artifacts are present in 'models/' and 'data/processed/'.")
