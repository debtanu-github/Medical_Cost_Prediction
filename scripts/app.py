import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Get the absolute path to the models directory
current_dir = os.path.dirname(__file__)
models_dir = os.path.join(os.path.dirname(current_dir), 'models')

# Load model and scaler
try:
    model = joblib.load(os.path.join(models_dir, 'medical_cost_model.joblib'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
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
        input_data['sex'] = (input_data['sex'] == 'male').astype(int)  # male=1, female=0
        input_data['smoker_yes'] = (input_data['smoker'] == 'yes').astype(int)
        
        # Create dummy variables for region
        region_dummies = pd.get_dummies(input_data['region'], prefix='region')
        input_data = pd.concat([input_data, region_dummies], axis=1)
        
        # Select only the features that the model was trained on
        model_features = ['age', 'bmi', 'smoker_yes']  # Update this if model changes
        input_data_model = input_data[model_features]
        
        # Scale the input
        input_scaled = scaler.transform(input_data_model)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Convert back from log scale and format as float
        cost = float(np.exp(prediction[0]) - 1)
        
        # Display prediction with formatted currency
        st.success(f"Predicted Insurance Cost: ${cost:,.2f}")
        
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

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit")
