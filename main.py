import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Crear carpetas necesarias
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

def generate_synthetic_data(n_samples=1000, seed=42):
    """Genera datos realistas basados en patrones de transporte cubano"""
    np.random.seed(seed)
    data = {
        'hora': np.random.uniform(6.0, 22.0, n_samples),
        'dia_semana': np.random.randint(0, 7, n_samples),  # 0=Lun ... 6=Dom
        'es_feriado': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'temperatura': np.random.normal(28, 3, n_samples).clip(20, 35),
        'lluvia_mm': np.random.exponential(2, n_samples),
        'nivel_llenado': np.random.randint(1, 6, n_samples),
    }
    df = pd.DataFrame(data)

    # Regla lógica para simular retraso (>15 min)
    prob_retraso = (
        0.15 * (df['hora'].between(7, 9) | df['hora'].between(17, 19)).astype(int) +
        0.10 * (df['dia_semana'] < 5).astype(int) +
        0.25 * (df['lluvia_mm'] > 5).astype(int) +
        0.20 * (df['nivel_llenado'] >= 4).astype(int) +
        0.05 * df['es_feriado']
    )
    df['retraso'] = np.random.binomial(1, np.clip(prob_retraso, 0, 1))
    return df

def main():
    print("📊 1. Generando y guardando datos...")
    df = generate_synthetic_data()
    df.to_csv('data/transporte_cuba.csv', index=False)

    X = df.drop('retraso', axis=1)
    y = df['retraso']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("🔧 2. Configurando pipeline de ML...")
    numeric_features = ['hora', 'temperatura', 'lluvia_mm', 'nivel_llenado']
    categorical_features = ['dia_semana']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    print("🎓 3. Entrenando modelo...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n📈 MÉTRICAS DE EVALUACIÓN:")
    print(classification_report(y_test, y_pred))
    print(f"🎯 Accuracy General: {accuracy_score(y_test, y_pred):.3f}")

    print("💾 4. Guardando modelo...")
    joblib.dump(model, 'models/modelo_transporte.pkl')

    # Extraer importancia de variables
    ohe = model.named_steps['preprocessor'].named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_names) + ['es_feriado']

    rf = model.named_steps['classifier']
    importances = pd.Series(rf.feature_importances_, index=all_feature_names).sort_values(ascending=False)

    print("\n🔍 TOP 5 VARIABLES MÁS IMPORTANTES:")
    print(importances.head())

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.index, y=importances.values, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('Importancia de Variables en la Predicción de Retrasos')
    plt.ylabel('Importancia Relativa')
    plt.tight_layout()
    plt.savefig('models/importancia_variables.png', dpi=150)
    print("\n✅ ¡Pipeline completado! Modelo y gráfico guardados en /models")

if __name__ == '__main__':
    main()