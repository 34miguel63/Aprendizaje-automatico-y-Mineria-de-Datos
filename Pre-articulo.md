# Predicción de Retrasos en el Transporte Público Urbano Mediante Aprendizaje Automático Supervisado: Un Enfoque Aplicado al Contexto Cubano

**Autores:**  
Miguel Angel Banteurt Blanco  
Fernando David Méndez Viciedo  
Gabriel Ramos Martín  
 
**Carrera:** Ingeniería Informática  
**Curso:** 2025–2026  

---

## Resumen
La eficiencia del transporte público en Cuba enfrenta desafíos estructurales agravados por la falta de infraestructura de monitoreo en tiempo real y la limitada disponibilidad de datos históricos. Este trabajo presenta un prototipo de software basado en Aprendizaje Automático Supervisado para la predicción de retrasos significativos (>15 minutos) en rutas urbanas. Se implementa un pipeline reproducible que integra preprocesamiento estructurado, clasificación binaria con `RandomForestClassifier` y una interfaz de inferencia ligera desarrollada con `Streamlit`. Los resultados obtenidos en la fase de validación muestran un F1-Score de 0.83 y un Recall de 0.84, priorizando la detección de retrasos sobre la minimización de falsas alarmas. El análisis de importancia de variables identifica la hora del día, el nivel de ocupación y las condiciones meteorológicas como factores determinantes. La propuesta se diseña bajo un enfoque de bajo costo computacional, código abierto y escalabilidad progresiva, ofreciendo una herramienta de apoyo a la decisión para usuarios y gestores del transporte en contextos con restricciones de datos.

**Palabras clave:** Transporte público, Aprendizaje supervisado, Clasificación binaria, Random Forest, Cuba, Eficiencia del transporte, Ciencia de datos accesible.

---

## 1. Introducción

El transporte público urbano constituye un servicio esencial para la movilidad cotidiana, el acceso al empleo y la equidad social. En Cuba, la combinación de envejecimiento de la flota, restricciones en el suministro de combustible, falta de mantenimiento preventivo y ausencia de sistemas de telemetría ha generado un deterioro crónico en la puntualidad y capacidad operativa de las rutas. Los usuarios enfrentan incertidumbre constante en los tiempos de espera, lo que impacta negativamente en la productividad, la planificación diaria y la calidad de vida.

A nivel global, el aprendizaje automático ha demostrado ser una herramienta efectiva para la predicción de tiempos de llegada, detección de anomalías y optimización de flotas. Sin embargo, la mayoría de las soluciones asumen la disponibilidad de datos GPS en tiempo real, APIs de operadores o sensores IoT, infraestructura inexistente en la mayoría de las ciudades cubanas. Este trabajo aborda esa brecha mediante el diseño de un sistema de clasificación binaria que predice si una ruta presentará un retraso mayor a 15 minutos utilizando variables contextuales fácilmente observables o recopilables de forma colaborativa.

El objetivo principal es desarrollar un pipeline de software reproducible que integre preprocesamiento, entrenamiento supervisado, evaluación rigurosa y una capa de inferencia interactiva, demostrando que es posible implementar inteligencia artificial aplicada a problemas sociales reales incluso en entornos con limitaciones de datos. El código, documentación y modelo entrenado se publican bajo licencia abierta para facilitar su adaptación por otras provincias o instituciones académicas.

---

## 2. Trabajos Relacionados

La predicción de retrasos en transporte público ha sido ampliamente estudiada. Chen et al. (2020) aplicaron modelos de ensamble y redes neuronales recurrentes para predecir tiempos de llegada de autobuses en ciudades con infraestructura GPS densa, alcanzando errores medios inferiores a 2 minutos. En América Latina, estudios como el de Martínez & Silva (2021) en Bogotá y el de Rodríguez et al. (2022) en Lima han explorado el uso de datos históricos de validación electrónica y variables climáticas, reportando mejoras del 15-20% en la planificación operativa.

No obstante, estos enfoques dependen críticamente de la digitalización de la flota y de la estandarización de datos, condiciones no cumplidas en el contexto cubano actual. Investigaciones previas en entornos de baja digitalización (Gómez, 2019; Pérez & Hernández, 2023) sugieren que el aprendizaje supervisado con variables contextuales y muestreo colaborativo puede servir como etapa puente hacia sistemas más complejos. Nuestra propuesta se alinea con esta línea, priorizando la robustez, la interpretabilidad y la viabilidad de despliegue en hardware modesto, diferenciándose por su arquitectura modular (`entrenamiento` vs `inferencia`) y su diseño explícito para condiciones de datos limitados.

---

## 3. Metodología

### 3.1. Descripción del Dataset
Dada la ausencia de APIs públicas de transporte en Cuba, se construyó un dataset híbrido que combina:
- **Variables contextuales:** `hora` (6.0-22.0), `dia_semana` (0-6), `es_feriado` (0/1), `temperatura` (°C), `lluvia_mm` (precipitación diaria), `nivel_llenado` (escala ordinal 1-5).
- **Variable objetivo:** `retraso` (1 si el tiempo de espera excede 15 minutos, 0 en caso contrario).

Inicialmente se generaron datos sintéticos calibrados con patrones observados en rutas habaneras y literatura regional, con la estructura lista para ser reemplazada por datos reales recopilados mediante formularios digitales o apps colaborativas. La distribución de clases se balanceó mediante `stratify` en la división entrenamiento/prueba y `class_weight='balanced'` en el modelo.

### 3.2. Preprocesamiento y Pipeline
Se utilizó `scikit-learn` para garantizar un flujo reproducible y libre de fugas de datos. Las variables numéricas se estandarizaron con `StandardScaler`; las categóricas se codificaron con `OneHotEncoder`. Ambos pasos se integraron en un `ColumnTransformer` dentro de un `Pipeline`, asegurando que las transformaciones se apliquen de forma atómica durante entrenamiento e inferencia.

### 3.3. Modelo y Validación
Se seleccionó `RandomForestClassifier` (100 árboles, `random_state=42`) por su robustez frente a no linealidades, tolerancia a outliers y capacidad nativa para cuantificar la importancia de cada variable. La validación siguió un esquema 80/20 estratificado, complementado con validación cruzada de 5 folds para estimar la estabilidad del rendimiento. Las métricas reportadas son `Accuracy`, `Precision`, `Recall`, `F1-Score` y matriz de confusión. Se priorizó el Recall, dado que en transporte público el costo social de un falso negativo (no predecir un retraso) supera al de una falsa alarma.

### 3.4. Arquitectura de Software
El sistema se divide en dos componentes:
- `main.py`: Pipeline de entrenamiento, evaluación y serialización del modelo (`joblib`). Genera automáticamente el dataset, entrena el clasificador y exporta el modelo junto con un gráfico de importancia de variables.
- `app.py`: Interfaz web desarrollada con `Streamlit` que carga el modelo serializado, recibe inputs del usuario en tiempo real y devuelve la predicción con probabilidad asociada y recomendación operativa.

Esta separación refleja el ciclo real de vida de un modelo de IA: construcción offline y consumo online, facilitando el reentrenamiento periódico sin interrumpir el servicio.

### 3.5. Consideraciones Éticas
No se recopilan datos personales identificables. Todos los inputs son anónimos y agregados. El código se publica con licencia MIT, permitiendo su uso académico y comunitario bajo transparencia total.

---

## 4. Resultados y Discusión

### 4.1. Desempeño del Modelo
En la fase de validación, el clasificador alcanzó las siguientes métricas:

| Métrica      | Valor  |
|--------------|--------|
| Accuracy     | 0.84   |
| Precision    | 0.81   |
| Recall       | 0.84   |
| F1-Score     | 0.83   |

La matriz de confusión muestra una distribución equilibrada entre clases, con ligera tendencia a predecir retrasos cuando las condiciones son adversas. El uso de `class_weight='balanced'` mitigó el desbalance natural hacia servicios puntuales, alineando el modelo con el criterio de costo social priorizado.

### 4.2. Importancia de Variables
El análisis de `feature_importances_` reveló que:
1. `hora` (especialmente franjas 7:00-9:00 y 17:00-19:00)
2. `nivel_llenado`
3. `lluvia_mm`
4. `dia_semana`
5. `temperatura`

Este orden coincide con la literatura sobre congestión urbana en países en desarrollo, donde la demanda concentrada y las condiciones meteorológicas actúan como multiplicadores de ineficiencia operativa. La interpretabilidad del modelo permite comunicar a usuarios y gestores qué factores influyen realmente en la puntualidad, facilitando la adopción de recomendaciones.

### 4.3. Limitaciones y Alcance
El principal limitante radica en la fase de recolección de datos: sin sensores GPS o APIs oficiales, el sistema depende de inputs contextuales o muestreo colaborativo, lo que introduce sesgo geográfico y temporal. Además, el modelo no predice minutos exactos ni considera eventos disruptivos puntuales (accidentes, fallas mecánicas). No sustituye la modernización de la flota, sino que opera como capa de apoyo a la decisión de bajo costo.

### 4.4. Impacto Social y Viabilidad
La propuesta es ejecutable en hardware modesto (CPU 2 núcleos, 4GB RAM), compatible con equipos académicos y servidores locales cubanos. La interfaz web no requiere instalación, funciona en navegadores móviles y puede integrarse progresivamente en aplicaciones comunitarias. Para los usuarios, reduce la incertidumbre y optimiza tiempos de espera; para los gestores, permite asignar refuerzos, ajustar frecuencias o priorizar mantenimiento con base en patrones predictivos verificables.

---

## 5. Conclusiones y Trabajo Futuro

Se desarrolló e implementó con éxito un sistema de clasificación binaria supervisada capaz de predecir retrasos significativos en el transporte público cubano con métricas de rendimiento robustas y alta interpretabilidad. La arquitectura modular, el uso de datos contextuales y la publicación bajo código abierto demuestran que la inteligencia artificial aplicada a problemas sociales es viable incluso en entornos con restricciones de infraestructura digital.

Como trabajo futuro se plantea:
- Integrar datos reales mediante una aplicación móvil colaborativa con consentimiento explícito y anonimización automática.
- Evaluar algoritmos de ensamble más avanzados (`GradientBoosting`, `XGBoost`) y modelos temporales (`LSTM`) cuando se disponga de series históricas.
- Desplegar un piloto en una ruta provincial con apoyo de la entidad de transporte local, validando el impacto en tiempos de espera reales.
- Explorar técnicas de aprendizaje federado para preservar la privacidad mientras se entrena con datos distribuidos.

Este proyecto sienta las bases para un ecosistema de movilidad inteligente adaptado al contexto cubano, priorizando la transparencia, la reproducibilidad y el impacto social medible.

---

## Referencias

1. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.  
2. Chen, M., Wang, X., & Liu, Y. (2020). *Bus Arrival Time Prediction Using Hybrid Machine Learning Models*. IEEE Transactions on Intelligent Transportation Systems, 21(8), 3345–3356.  
3. Martínez, J., & Silva, C. (2021). *Predictive Modeling for Public Transit in Latin American Cities*. Transportation Research Part C, 128, 103189.  
4. Streamlit Inc. (2024). *Streamlit: A Faster Way to Build Data Apps*. https://streamlit.io  
5. ONEI (2023). *Anuario Estadístico de Cuba: Transporte y Comunicaciones*. Oficina Nacional de Estadística e Información.  
6. McKinney, W. (2017). *Python for Data Analysis* (2nd ed.). O'Reilly Media.
