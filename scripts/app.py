import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Medical Cost Estimator", layout="wide")

# --- Configuration & Model Loading ---
try:
    current_dir = Path(__file__).parent
except NameError: # Fallback for interactive environments
    current_dir = Path.cwd()

# Assuming app.py is in 'scripts' and 'models' is one level up
models_dir = current_dir.parent / 'models'
CHOSEN_MODEL_FILENAME = 'random_forest_untuned_medical_cost_model.joblib' # Your RF model
MODEL_PATH = str(models_dir / CHOSEN_MODEL_FILENAME)

USD_TO_INR = 80 # Example fixed rate

@st.cache_resource # Cache the loaded model
def load_model(model_path_func_arg):
    try:
        model = joblib.load(model_path_func_arg)
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path_func_arg}.")
        # In a real app, you might want to log this or handle it more gracefully
        # For now, returning None and checking later.
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

rf_model = load_model(MODEL_PATH) # This will be your Random Forest model

# --- Define expected features for the Random Forest model (after OHE) ---
# This list MUST match the columns (and their order) used for training rf_model,
# which are the columns in tree_data.csv excluding 'charges'.
EXPECTED_RF_FEATURES = [
    'age', 'bmi', 'children', 'sex_male', 'smoker_yes',
    'region_northwest', 'region_southeast', 'region_southwest'
]

# --- Performance Metrics (Hardcode these from your notebook) ---
# For the CHOSEN MODEL (Random Forest)
chosen_model_name_for_info = "Random Forest"
# MSE for RF = (RMSE_rf)^2 = (4576.30)^2 = 20942527.69
chosen_model_metrics = {
    "MAE": "$2,550.08",
    "MSE": "20,942,527.69", # Displaying as string, you can format if needed
    "R² Score": "0.8651"
}

# For OTHER MODELS (Linear Regression)
# !!! IMPORTANT: REPLACE THESE PLACEHOLDERS WITH YOUR ACTUAL CALCULATED VALUES !!!
# These values should be from evaluating your LR model on the ORIGINAL DOLLAR SCALE.
other_models_performance = [
    {
        "name": "Linear Regression (Log Target, 3 Features)",
        "MAE": "$3,900.00",  # !!! Plausible Placeholder - REPLACE !!!
        "MSE": "38,000,000.00", # !!! Plausible Placeholder - REPLACE !!! (e.g., (sqrt(MSE_log_scale) * some_factor)^2, then back-transformed)
        "R² Score": "0.7600"  # !!! Plausible Placeholder - REPLACE !!!
    }
]

# --- Streamlit UI ---
st.title("🩺 Medical Insurance Cost Prediction")
st.write("Enter your details below to predict your estimated annual insurance cost.")

# Input form using your preferred layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        sex_input = st.selectbox("Sex", ["male", "female"]) # Renamed to avoid conflict
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=20.0)
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker_input = st.selectbox("Smoker", ["yes", "no"]) # Renamed to avoid conflict
        region_input = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"]) # Renamed
    submit_button = st.form_submit_button("Predict Cost")

# --- Prediction and Results Display ---
if rf_model is None: # Check if model loading failed
    st.error(f"Critical Error: The prediction model could not be loaded from {MODEL_PATH}. Please check the file path and ensure the model exists in the '{models_dir.name}' directory.")
else:
    if submit_button:
        try:
            # 1. Create a dictionary for input features (numerical first)
            input_features_dict = {
                'age': age,
                'bmi': bmi,
                'children': children
            }

            # 2. Perform One-Hot Encoding manually for Random Forest
            # Assumes pd.get_dummies(..., drop_first=True) was used for tree_data.csv
            # 'female' dropped for sex, 'no' dropped for smoker, 'northeast' dropped for region
            input_features_dict['sex_male'] = 1 if sex_input == 'male' else 0
            input_features_dict['smoker_yes'] = 1 if smoker_input == 'yes' else 0
            input_features_dict['region_northwest'] = 1 if region_input == 'northwest' else 0
            input_features_dict['region_southeast'] = 1 if region_input == 'southeast' else 0
            input_features_dict['region_southwest'] = 1 if region_input == 'southwest' else 0

            # 3. Create a DataFrame in the expected order of features
            input_df = pd.DataFrame([input_features_dict])
            input_processed_rf = input_df.reindex(columns=EXPECTED_RF_FEATURES, fill_value=0)

            # 4. Make Prediction with Random Forest
            cost_usd = rf_model.predict(input_processed_rf)[0]
            cost_inr = cost_usd * USD_TO_INR

            # Display predictions
            st.subheader("Predicted Insurance Cost:")
            pred_col1, pred_col2 = st.columns(2)
            with pred_col1:
                st.metric("USD", f"${cost_usd:,.2f}")
            with pred_col2:
                st.metric("INR", f"₹{cost_inr:,.2f}")

            if smoker_input == "yes":
                st.warning("Being a smoker significantly increases insurance costs!")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.write("Debug info - Processed Input Data Sent to RF Model:")
            if 'input_processed_rf' in locals():
                st.dataframe(input_processed_rf.head())
                st.write("Expected columns by RF model:", EXPECTED_RF_FEATURES)
            else:
                st.write("Input data processing failed before reaching the model.")

# --- Model Performance Information (Expander Section) ---
st.markdown("---")
with st.expander("About the Prediction Model & Performance", expanded=False):
    st.markdown(f"The prediction above is generated by a **{chosen_model_name_for_info}** model.")
    st.markdown(f"**Performance on Test Data ({chosen_model_name_for_info}):**")
    c1, c2, c3 = st.columns(3)
    c1.info(f"MAE: {chosen_model_metrics['MAE']}")
    c2.info(f"MSE: {chosen_model_metrics['MSE']}") # Changed from RMSE
    c3.info(f"R² Score: {chosen_model_metrics['R² Score']}")
    st.caption("MAE (Mean Absolute Error), MSE (Mean Squared Error). R² Score indicates variance explained.")

    st.markdown("<br>**Other Models Explored:**", unsafe_allow_html=True)
    if other_models_performance:
        for model_info in other_models_performance:
            st.markdown(f"*{model_info['name']}*")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.write(f"  - MAE: {model_info['MAE']}")
            m_col2.write(f"  - MSE: {model_info['MSE']}") # Changed from RMSE
            m_col3.write(f"  - R² Score: {model_info['R² Score']}")
    else:
        st.write("Details for other models are not currently displayed.")
    st.caption("Exploring multiple models helps in selecting a robust approach for prediction.")

# --- Footer and Additional Information ---
st.markdown("---")
with st.expander("About BMI Categories", expanded=False):
    st.markdown("""
    **Body Mass Index (BMI) ranges:**
    - Underweight: < 18.5
    - Normal weight: 18.5 - 24.9
    - Overweight: 25 - 29.9
    - Obese: ≥ 30
    """)
with st.expander("Currency Conversion Note", expanded=False):
    st.write(f"1 USD = ₹{USD_TO_INR:.2f} (Approximate fixed rate)")
st.markdown("<hr><p style='text-align: center; color: grey;'>Disclaimer: This is a demonstration project. Predictions are estimates and not financial advice.</p>", unsafe_allow_html=True)