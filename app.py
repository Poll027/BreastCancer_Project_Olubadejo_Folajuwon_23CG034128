import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="⚕️",
    layout="centered"
)

# Load Model and Scaler
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model/breast_cancer_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'breast_cancer_model.pkl' and 'scaler.pkl' are in the 'model' directory.")
        return None, None

model, scaler = load_models()

# Header
st.title("Breast Cancer Prediction System")
st.markdown("**Name:** Olubadejo Folajuwon | **Matric Number:** 23CG034128")
st.markdown("---")
st.write("Enter the tumor features below to predict if it is Benign or Malignant.")

# Input Features
st.subheader("Input Features")
col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", min_value=0.0, value=15.0, format="%.4f")
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=90.0, format="%.4f")
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1, format="%.4f")

with col2:
    texture_mean = st.number_input("Texture Mean", min_value=0.0, value=20.0, format="%.4f")
    area_mean = st.number_input("Area Mean", min_value=0.0, value=700.0, format="%.4f")

# Prediction
if st.button("Predict Diagnosis"):
    if model and scaler:
        # Prepare input array
        input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        prediction_prob = model.predict_proba(input_scaled)
        
        # Display Result
        st.markdown("---")
        if prediction[0] == 0: # Assuming 0 is Malignant based on sklearn dataset, but let's be careful. 
             # Actually sklearn dataset: 0 = Malignant, 1 = Benign.
             # Wait, usually 1 is the 'positive' class (the disease).
             # Let's verify standard sklearn behavior:
             # target_names: ['malignant', 'benign'] -> 0 is malignant, 1 is benign.
             # So if prediction is 0 -> Malignant.
            st.error(f"## Prediction: Malignant")
            st.write(f"Confidence: {prediction_prob[0][0]:.2%}")
        else:
            st.success(f"## Prediction: Benign")
            st.write(f"Confidence: {prediction_prob[0][1]:.2%}")
            
        st.info("Note: This system is strictly for educational purposes and must not be presented as a medical diagnostic tool.")
    else:
        st.error("Model could not be loaded.")

# Footer
st.markdown("---")
st.caption("Project 5 – Breast Cancer Prediction System")
