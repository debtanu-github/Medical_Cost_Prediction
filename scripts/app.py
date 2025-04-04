#this frontend is made by using streamlit and ai assistant
# will change it later to a proper frontend, lets just focus on building the model first

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Get the absolute path to the models directory
current_dir = Path(__file__).parent
models_dir = current_dir.parent / 'models'

# USD to INR conversion rate
USD_TO_INR = 80  # Example fixed rate

# Load model and scaler
try:
    model = joblib.load(str(models_dir / 'medical_cost_model.joblib'))
    scaler = joblib.load(str(models_dir / 'scaler.joblib'))
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.write("Current directory:", current_dir)
    st.write("Looking for model in:", models_dir)
    st.stop()

# Set page config
st.set_page_config(page_title="Medical Cost Predictor", layout="wide")

# Title and description
st.title("Medical Insurance Cost Prediction")
st.write("Enter your details below to predict insurance cost")

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=20.0)
    
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    submit = st.form_submit_button("Predict Cost")

if submit:
    try:
        # Prepare input data with all possible features
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })
        
        # Convert categorical variables
        input_data['sex'] = (input_data['sex'] == 'male').astype(int)
        input_data['smoker_yes'] = (input_data['smoker'] == 'yes').astype(int)
        
        # Create dummy variables for region
        region_dummies = pd.get_dummies(input_data['region'], prefix='region')
        input_data = pd.concat([input_data, region_dummies], axis=1)
        
        # Select only the features that the model was trained on
        model_features = ['age', 'bmi', 'smoker_yes']
        input_data_model = input_data[model_features]
        
        # Scale the input
        input_scaled = scaler.transform(input_data_model)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Convert back from log scale and format as float
        cost_usd = float(np.exp(prediction[0]) - 1)
        cost_inr = cost_usd * USD_TO_INR
        
        # Display predictions with formatted currency
        st.success("Predicted Insurance Cost:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("USD", f"${cost_usd:,.2f}")
        with col2:
            st.metric("INR", f"₹{cost_inr:,.2f}")
        
        # Add some context
        if smoker == "yes":
            st.warning("Being a smoker significantly increases insurance costs!")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Debug info:")
        st.write("Input data:")
        st.write(input_data)
        st.write("Model features used:", model_features)
        st.write("Scaled data shape:", input_scaled.shape if 'input_scaled' in locals() else "Not available")

# Add some helpful information
with st.expander("About BMI"):
    st.write("""
    BMI (Body Mass Index) ranges:
    - Underweight: < 18.5
    - Normal weight: 18.5 - 24.9
    - Overweight: 25 - 29.9
    - Obese: ≥ 30
    """)

# Add information about currency conversion
with st.expander("About Currency Conversion"):
    st.write(f"""
    Currency conversion is based on a fixed rate:
    - 1 USD = ₹{USD_TO_INR:.2f}
    - Note: Actual rates may vary. This is an approximate conversion.
    """)

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit") 