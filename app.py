import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Transporte Cuba IA", page_icon="🚌", layout="centered")
st.title("🚌 Predicción de Eficiencia del Transporte Público")
st.markdown("Ingresa las condiciones actuales para estimar si habrá retraso (>15 min).")

@st.cache_resource
def load_model():
    return joblib.load("models/modelo_transporte.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Modelo no encontrado. Ejecuta primero `main.py`")
    st.stop()

# Interfaz
col1, col2 = st.columns(2)
with col1:
    hora = st.slider("Hora del día", 6.0, 22.0, 8.5, step=0.5)
    dia_semana = st.selectbox("Día", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
with col2:
    temperatura = st.slider("Temperatura (°C)", 20, 35, 28)
    lluvia = st.number_input("Lluvia hoy (mm)", 0.0, 50.0, 0.0, step=0.5)
    nivel_llenado = st.slider("Nivel de llenado (1=Vacío, 5=Sobrecargado)", 1, 5, 3)

dia_map = {"Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3, "Viernes": 4, "Sábado": 5, "Domingo": 6}

if st.button("🔍 Predecir", type="primary", use_container_width=True):
    input_df = pd.DataFrame({
        'hora': [hora],
        'dia_semana': [dia_map[dia_semana]],
        'es_feriado': [0],
        'temperatura': [temperatura],
        'lluvia_mm': [lluvia],
        'nivel_llenado': [nivel_llenado]
    })

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ **Retraso probable** (Confianza: {prob:.1%})")
        st.info("💡 Recomendación: Sal con 20-30 min de anticipación o consulta rutas alternativas.")
    else:
        st.success(f"✅ **Servicio a tiempo** (Riesgo de retraso: {prob:.1%})")
        st.info("🟢 Condiciones favorables para viajar.")