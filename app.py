import streamlit as st
import pandas as pd
import pickle
import os
import time

# Page Config
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fun & Cool CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Vibrant Gradient Button */
    .stButton>button {
        width: 100%;
        border-radius: 30px;
        height: 3.5em;
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        font-weight: 600;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 239, 125, 0.6);
        background: linear-gradient(90deg, #38ef7d 0%, #11998e 100%);
    }

    /* Glassmorphic Result Card */
    .result-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 30px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .metric-value {
        font-size: 3.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(#11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 10px;
    }
    
    /* Dark Mode Adjustments */
    @media (prefers-color-scheme: dark) {
        .metric-label { color: #ccc; }
        .result-card { background: rgba(0, 0, 0, 0.2); }
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌽 FutureFarms AI")
st.markdown("### 🚀 Supercharge your harvest with AI prediction.")
st.markdown("---")

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    # Paths
    model_path = "models/best_model.pkl"
    preprocessor_path = "data/processed/preprocessor.pkl"
    columns_path = "data/processed/columns.pkl"
    data_path = "data/interim/cleaned_data.csv"
    
    if not os.path.exists(model_path):
        st.error("⚠️ Model not found. Please run training first.")
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
    st.sidebar.image("https://img.icons8.com/color/96/000000/tractor.png", width=80)
    st.sidebar.header("🌾 Farm Settings")
    st.sidebar.info(
        "Tweak the parameters to simulate different farming scenarios and see how AI predicts your yield!"
    )
    st.sidebar.markdown("---")
    
    # --- Input Form ---
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("🌍 Geography & Crop")
        
        # State
        states = sorted(df['state'].unique())
        selected_state = st.selectbox("📍 State", states)
        
        # Season
        seasons = sorted(df['season'].unique())
        selected_season = st.selectbox("🌤️ Season", seasons)
        
        # Crop
        crops = sorted(df['crop'].unique())
        selected_crop = st.selectbox("🌱 Crop Type", crops)
        
    with col2:
        st.subheader("📊 Farm Metrics")
        
        # Rainfall
        avg_rainfall = df['annual_rainfall'].mean()
        annual_rainfall = st.number_input(
            "🌧️ Annual Rainfall (mm)", 
            min_value=0.0, 
            value=float(avg_rainfall),
            step=10.0,
            help="Enter the total annual rainfall in millimeters."
        )
        
        # Pesticide
        avg_pesticide = df['pesticide'].mean()
        pesticide = st.number_input(
            "🧪 Pesticide Usage (kg)", 
            min_value=0.0, 
            value=float(avg_pesticide),
            step=0.1,
            help="Enter the amount of pesticide used in kilograms."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Prediction ---
    # Center the button
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        predict_btn = st.button("✨ GENERATE PREDICTION ✨")

    if predict_btn:
        # Fun spinner
        with st.spinner("🤖 AI is crunching the numbers..."):
            time.sleep(1) # Fake delay for suspense
            
            # Create DataFrame
            input_data = pd.DataFrame({
                'annual_rainfall': [annual_rainfall],
                'pesticide': [pesticide],
                'crop': [selected_crop],
                'season': [selected_season],
                'state': [selected_state]
            })
            
            # One-Hot Encoding
            input_encoded = pd.get_dummies(input_data, columns=['crop', 'season', 'state'], drop_first=True)
            
            # Align with model columns
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
            
            # Bool to Int conversion
            input_encoded = input_encoded.astype(int)
            
            try:
                input_scaled = preprocessor.transform(input_encoded)
                
                # Convert back to DataFrame with column names to silence warning
                input_scaled_df = pd.DataFrame(input_scaled, columns=model_columns)
                
                prediction = model.predict(input_scaled_df)
                
                # Success balloons
                st.balloons()
                
                # Display Results
                st.markdown(f"""
                <div class="result-card">
                    <p class="metric-label">Estimated Harvest Yield</p>
                    <p class="metric-value">{prediction[0]:.2f}</p>
                    <p style="margin-top: 10px; font-weight: 600; color: #555;">Units per Hectare</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Contextual Toast
                if prediction[0] > df['yield'].mean():
                    st.toast("🎉 Wow! That's a high yield prediction!", icon="🌟")
                else:
                    st.toast("💡 Tip: Try optimizing rainfall or pesticide levels.", icon="📈")
                
            except Exception as e:
                st.error(f"💥 Oops! Something went wrong: {e}")

else:
    st.warning("Please ensure all model artifacts are present in 'models/' and 'data/processed/'.")
