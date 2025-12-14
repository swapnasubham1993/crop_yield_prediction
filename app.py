import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page config
st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

# Load Assets
@st.cache_resource
def load_artifacts():
    try:
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('data/processed/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('data/processed/columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
        return model, preprocessor, model_columns
    except FileNotFoundError as e:
        st.error(f"Artifacts not found: {e}. Please ensure the DVC pipeline has run.")
        return None, None, None

@st.cache_data
def load_data():
    try:
        # Load only necessary columns for dropdowns from raw data
        # Check params.yaml for raw path usually, but hardcoding for app simplicity or loading from config
        df = pd.read_csv('data/raw/crop_yield.csv', usecols=['Crop', 'Season', 'State'])
        return df
    except FileNotFoundError:
        st.error("data/raw/crop_yield.csv not found.")
        return None

model, preprocessor, model_columns = load_artifacts()
data = load_data()

# Header
st.title("🌾 Crop Yield Prediction System")
st.markdown("""
This application uses a Machine Learning model (Gradient Boosting Regressor) to predict crop yield based on various agricultural parameters.
Please fill in the details below to get an estimated yield.
""")

if model is not None and data is not None:
    # Sidebar for inputs? Or Main area. Using Main area with columns as per plan.
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crop Details")
        crop_list = sorted(data['Crop'].unique().tolist())
        selected_crop = st.selectbox("Select Crop", crop_list)
        
        season_list = sorted(data['Season'].unique().tolist())
        selected_season = st.selectbox("Select Season", season_list)
        
        state_list = sorted(data['State'].unique().tolist())
        selected_state = st.selectbox("Select State", state_list)
        
        # area = st.number_input("Area (Hectares)", min_value=0.01, value=100.0, step=10.0) # Removed per user request

    with col2:
        st.subheader("Agricultural Factors")
        # Production is conceptually an output, but required as feature for this specific model
        # production = st.number_input("Production (Tonnes)", min_value=0.0, value=500.0, step=50.0, 
        #                              help="Total production amount. Note: This model was trained using production as a feature.") # Removed per user request
        
        annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1000.0, step=50.0)
        
        # fertilizer = st.number_input("Fertilizer (kg)", min_value=0.0, value=5000.0, step=100.0) # Removed per user request
        
        pesticide = st.number_input("Pesticide (kg)", min_value=0.0, value=100.0, step=10.0)

    # Prediction Logic
    if st.button("Predict Yield", type="primary"):
        # Create dataframe from input
        input_data = {
            'Crop': [selected_crop],
            'Season': [selected_season],
            'State': [selected_state],
            # 'Area': [area],
            # 'Production': [production],
            'Annual_Rainfall': [annual_rainfall],
            # 'Fertilizer': [fertilizer],
            'Pesticide': [pesticide]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # One-hot encode
        # We need to ensure we create the same dummy structure. 
        # Using get_dummies on single row might miss columns, so we align with model_columns
        
        category_cols = input_df.select_dtypes(include=['object']).columns
        input_encoded = pd.get_dummies(input_df, columns=category_cols, drop_first=True)
        
        # Align columns
        # Create a dataframe with all model columns set to 0
        final_input = pd.DataFrame(columns=model_columns)
        
        # We need to construct a single row with 0s, then update with our calculated values
        # A safer way to handle alignment:
        
        # Initialize dictionary with 0 for all model columns
        input_dict = {col: 0 for col in model_columns}
        
        # Update with numeric values directly
        for col in ['Annual_Rainfall', 'Pesticide']:
            if col in input_dict:
                input_dict[col] = input_data[col][0]
        
        # Update with categorical matches
        # The column names in model_columns would look like 'Crop_Rice', 'State_Assam', etc.
        # Check generated dummy columns from input_df
        for col in input_encoded.columns:
            if col in input_dict:
                input_dict[col] = input_encoded[col][0]
        
        # Handling the case where get_dummies might produce columns NOT in model_columns (unlikely if drop_first matched)
        # But critically: The input is just one row.
        # If I selected 'Rice', get_dummies produces 'Crop_Rice'=1.
        # If 'Crop_Rice' is in model_columns, we set it to 1.
        # If I selected a category that was dropped (the base case) during training, no column is created/set, which is correct (all 0s).
        
        # NOTE: drop_first=True in training means one category per feature is implicitly represented by all 0s.
        # If input has that category, get_dummies(drop_first=True) on a single row might result in NO categorical columns if it happens to be the first one alphabetically?
        # WAIT. get_dummies on a Dataframe with 1 row and 1 unique value in a categorical column:
        # If drop_first=True, it might drop the *only* column it creates?
        # Example: Input Crop='Rice'. get_dummies(['Rice'], drop_first=True) -> Empty DataFrame?
        # Let's verify behavior. 
        # pd.get_dummies(pd.Series(['A']), drop_first=True) -> Empty DF.
        # This is a risk.
        
        # Robust approach for inference with drop_first=True models:
        # 1. We know the exact column names expected (model_columns).
        # 2. We can manually construct the feature vector.
        
        # Reset input_dict
        input_dict = {col: 0 for col in model_columns}
        
        # Fill numeric (Use lowercase keys to match training)
        # input_dict['area'] = area
        # input_dict['production'] = production
        input_dict['annual_rainfall'] = annual_rainfall
        # input_dict['fertilizer'] = fertilizer
        input_dict['pesticide'] = pesticide
        
        # Fill categorical
        # Preprocessing used lowercased columns and lowercased values.
        # e.g. 'Crop' -> 'crop', value 'Rice' -> 'rice'.
        # get_dummies creates 'crop_rice'.
        
        # Helper to match ingestion cleaning
        def clean_text(text):
            return text.lower().strip().replace(" ", "_")

        crop_key = f"crop_{clean_text(selected_crop)}"
        season_key = f"season_{clean_text(selected_season)}"
        state_key = f"state_{clean_text(selected_state)}"
        
        if crop_key in input_dict:
            input_dict[crop_key] = 1
        if season_key in input_dict:
            input_dict[season_key] = 1
        if state_key in input_dict:
            input_dict[state_key] = 1
            
        # Convert to DF
        final_df = pd.DataFrame([input_dict])
        
        # Scale
        try:
            X_scaled = preprocessor.transform(final_df)
            
            # Predict
            prediction = model.predict(X_scaled)[0]
            
            st.success(f"Predicted Yield: {prediction:.4f}")
            
            # Optional: Display input data used
            with st.expander("Show processed input data"):
                st.write(final_df)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("System not ready. Please check files.")
