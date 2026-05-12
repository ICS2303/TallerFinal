
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(layout="centered")

# Load the trained model
# Assuming the model is saved in the same directory as the Streamlit app or accessible path
# Using the Logistic Regression model for this application as it was favored for recall.
try:
    model = joblib.load('modelo_desercion_lr.pkl')
    st.success("Modelo de Regresión Logística cargado exitosamente!")
except FileNotFoundError:
    st.error("Error: modelo_desercion_lr.pkl no encontrado. Asegúrate de que el modelo esté guardado en la ubicación correcta.")
    st.stop() # Stop the app if model is not found

# Title of the application
st.title("🎓 Predicción de Deserción Estudiantil")
st.write("Ingresa los datos del estudiante para predecir si desertará o no.")

# Input fields for features
st.sidebar.header("Parámetros del Estudiante")

edad = st.sidebar.slider("Edad", min_value=18, max_value=25, value=20, step=1)
promedio = st.sidebar.slider("Promedio (0-100)", min_value=50.0, max_value=100.0, value=75.0, step=0.1)
asistencia = st.sidebar.selectbox("Asistencia", options=[0, 1], format_func=lambda x: "Buena (1)" if x == 1 else "Mala (0)")
horas_estudio = st.sidebar.slider("Horas de Estudio Semanales", min_value=0.0, max_value=40.0, value=15.0, step=0.5)
uso_plataforma = st.sidebar.selectbox("Uso de Plataforma Educativa", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
materias_perdidas = st.sidebar.slider("Materias Perdidas", min_value=0, max_value=3, value=1, step=1)
nivel_socioeconomico = st.sidebar.selectbox("Nivel Socioeconómico", options=[0, 1], format_func=lambda x: "Bajo (0)" if x == 0 else "Alto (1)")
trabaja = st.sidebar.selectbox("Trabaja", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
acceso_internet = st.sidebar.selectbox("Acceso a Internet", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")

# Create a DataFrame from inputs
input_data = pd.DataFrame([[edad, promedio, asistencia, horas_estudio, uso_plataforma, materias_perdidas, nivel_socioeconomico, trabaja, acceso_internet]],
                          columns=['edad', 'promedio', 'asistencia', 'horas_estudio', 'uso_plataforma', 'materias_perdidas', 'nivel_socioeconomico', 'trabaja', 'acceso_internet'])


st.subheader("Datos del Estudiante Ingresados:")
st.write(input_data)

# Make prediction
if st.button("Predecir Deserción"):
    if model is not None:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("Resultado de la Predicción:")
        if prediction[0] == 1:
            st.error(f"¡El estudiante tiene una alta probabilidad de desertar! ⚠️ (Probabilidad: {prediction_proba[0][1]:.2f})")
            st.markdown("Considera implementar intervenciones tempranas como tutorías o apoyo socioeconómico.")
        else:
            st.success(f"El estudiante tiene baja probabilidad de desertar. ✅ (Probabilidad: {prediction_proba[0][0]:.2f})")
            st.markdown("¡Excelente! El estudiante parece estar en buen camino.")

        st.markdown("--- Generado por Colab Composer ---")
    else:
        st.warning("El modelo no ha sido cargado. Por favor, verifica la ruta del archivo.")
