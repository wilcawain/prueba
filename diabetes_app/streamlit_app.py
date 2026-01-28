# app_diabetes.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Diabetes",
    page_icon="🩺",
    layout="wide"
)

# Título y descripción
st.title("🩺 Predictor de Diabetes")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning para predecir el riesgo de diabetes 
basándose en parámetros de salud del paciente.
""")

# Barra lateral para entrada de datos
st.sidebar.header("📋 Datos del Paciente")

def get_user_input():
    pregnancies = st.sidebar.slider('Embarazos', 0, 17, 1)
    glucose = st.sidebar.slider('Glucosa (mg/dL)', 50, 200, 100)
    blood_pressure = st.sidebar.slider('Presión Arterial (mm Hg)', 40, 140, 70)
    skin_thickness = st.sidebar.slider('Espesor de Piel (mm)', 10, 100, 30)
    insulin = st.sidebar.slider('Insulina (μU/mL)', 0, 900, 100)
    bmi = st.sidebar.slider('Índice de Masa Corporal', 15.0, 60.0, 25.0)
    dpf = st.sidebar.slider('Función del Pedigrí de Diabetes', 0.0, 2.5, 0.5)
    age = st.sidebar.slider('Edad', 15, 90, 30)
    
    return {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

# Obtener datos del usuario
user_data = get_user_input()

# Convertir a DataFrame
input_df = pd.DataFrame([user_data])

# Mostrar datos ingresados
st.subheader("📊 Datos Ingresados")
st.dataframe(input_df, use_container_width=True)

# Columnas para métricas
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Glucosa", f"{user_data['Glucose']} mg/dL", 
              "Alerta" if user_data['Glucose'] > 126 else "Normal")
with col2:
    st.metric("Presión Arterial", f"{user_data['BloodPressure']} mm Hg",
              "Alta" if user_data['BloodPressure'] > 90 else "Normal")
with col3:
    st.metric("IMC", f"{user_data['BMI']:.1f}",
              "Sobrepeso" if user_data['BMI'] > 25 else "Normal")
with col4:
    st.metric("Edad", user_data['Age'])

# Sección de predicción
st.subheader("🔮 Predicción")

# Cargar o entrenar modelo
@st.cache_resource
def load_model():
    # Cargar datos y entrenar modelo (en producción se cargaría un modelo pre-entrenado)
    url = "https://raw.githubusercontent.com/LuisPerezTimana/Webinars/main/diabetes.csv"
    df = pd.read_csv(url)
    
    # Preprocesamiento
    cols_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_con_ceros] = df[cols_con_ceros].replace(0, np.nan)
    for col in cols_con_ceros:
        df[col].fillna(df[col].median(), inplace=True)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X.columns.tolist()

# Cargar modelo
model, feature_names = load_model()

# Realizar predicción
prediction = model.predict(input_df)
probability = model.predict_proba(input_df)

# Mostrar resultados
result_col1, result_col2 = st.columns(2)

with result_col1:
    st.markdown("### Resultado")
    if prediction[0] == 1:
        st.error("⚠️ **Riesgo de Diabetes Detectado**")
        st.write("El modelo predice que existe riesgo de diabetes.")
    else:
        st.success("✅ **Sin Riesgo de Diabetes Detectado**")
        st.write("El modelo predice que no existe riesgo de diabetes.")

with result_col2:
    st.markdown("### Probabilidades")
    prob_df = pd.DataFrame({
        'Categoría': ['Sin Diabetes', 'Con Diabetes'],
        'Probabilidad': [probability[0][0] * 100, probability[0][1] * 100]
    })
    st.bar_chart(prob_df.set_index('Categoría'))

# Información adicional
st.subheader("📈 Explicación del Modelo")
st.markdown("""
**Características más importantes para la predicción:**
""")

# Importancia de características
feature_importance = pd.DataFrame({
    'Característica': feature_names,
    'Importancia': model.feature_importances_
}).sort_values('Importancia', ascending=False)

st.dataframe(feature_importance, use_container_width=True)

# Instrucciones para ejecutar
st.sidebar.subheader("🚀 Cómo Ejecutar")
st.sidebar.code("pip install streamlit pandas scikit-learn\nstreamlit run app_diabetes.py")

# Pie de página
st.markdown("---")
st.caption("""
*Nota: Esta aplicación es para fines educativos y de demostración. 
Para diagnóstico médico real, consulte siempre con un profesional de la salud.*
""")                          