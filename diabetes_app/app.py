import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

@st.cache_resource
def load_model():
    # Load the trained model and scaler
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model()

st.title("Diabetes Prediction Application")
st.write("Enter your health metrics below to get a prediction.")

if model is None or scaler is None:
    st.error("Model artifacts not found. Please run 'model_training.py' first to generate 'model.pkl' and 'scaler.pkl'.")
else:
    # Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
            glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        
        with col2:
            insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0, format="%.1f")
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
        
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        # Prepare input data
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        
        # Scale the input
        input_data_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_data_scaled)
        probability = model.predict_proba(input_data_scaled)
        
        # Display results
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"**Diabetic** (Probability: {probability[0][1]:.2%})")
            st.write("Please consult a healthcare professional for further advice.")
        else:
            st.success(f"**Not Diabetic** (Probability: {probability[0][0]:.2%})")
            st.write("Great! Maintain a healthy lifestyle.")

    # Sidebar info
    st.sidebar.header("About")
    st.sidebar.info("This application uses a Random Forest Classifier trained on the Pima Indians Diabetes Database.")
