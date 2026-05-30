# 🚌 Predicción de Retrasos en Transporte Público Urbano

> Proyecto académico de Inteligencia Artificial y Minería de Datos | 2do Año Ingeniería Informática  
> 🎯 **Objetivo:** Predecir si una ruta de transporte público presentará un retraso >15 min usando Aprendizaje Supervisado, en un contexto con limitaciones de infraestructura de datos.

---

## 👥 Autores
| Nombre | Rol | 
|--------|-----|
| **Miguel Ángel Banteurt Blanco** | Desarrollo de pipeline, entrenamiento y evaluación | 
| **Fernando David Méndez Viciedo** | Preprocesamiento, validación y métricas | 
| **Gabriel Ramos Martín** | Interfaz de usuario (Streamlit), documentación y despliegue | 

*Departamento de Ingeniería Informática FITIB*

---

## 📖 Descripción
La eficiencia del transporte público en Cuba enfrenta desafíos estructurales (flota envejecida, falta de telemetría, incertidumbre en tiempos de espera). Este proyecto implementa un **pipeline de Aprendizaje Supervisado** (Clasificación Binaria) que predice la probabilidad de retrasos significativos a partir de variables contextuales fácilmente observables: hora, día de la semana, condiciones meteorológicas y nivel de ocupación.

La arquitectura separa el **entrenamiento offline** (`main.py`) de la **inferencia online** (`app.py`), garantizando reproducibilidad, bajo consumo de recursos y escalabilidad progresiva hacia entornos con datos reales.

---

## ⚡ Características Principales
- ✅ Pipeline modular con `scikit-learn` (`ColumnTransformer` + `Pipeline`)
- ✅ Modelo interpretable: `RandomForestClassifier` con análisis de importancia de variables
- ✅ Interfaz web interactiva con `Streamlit` (sin necesidad de frontend adicional)
- ✅ Métricas de evaluación balanceadas (`Accuracy`, `Precision`, `Recall`, `F1-Score`)
- ✅ Código abierto, documentado y listo para ser adaptado a otras rutas/provincias

---

## 🛠️ Tecnologías Utilizadas
| Herramienta | Propósito |
|-------------|-----------|
| `Python 3.10+` | Lenguaje principal |
| `pandas`, `numpy` | Manipulación y generación de datos |
| `scikit-learn` | Preprocesamiento, modelo, validación y métricas |
| `matplotlib`, `seaborn` | Visualización y gráfica de importancia |
| `joblib` | Serialización eficiente del modelo |
| `streamlit` | Despliegue rápido de interfaz de predicción |

---

## 🚀 Instalación y Ejecución

### 1. Clonar y preparar entorno
```bash
git clone https://github.com/TU_USUARIO/transporte-ia-cuba.git
cd transporte-ia-cuba

# Crear entorno virtual (recomendado)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
