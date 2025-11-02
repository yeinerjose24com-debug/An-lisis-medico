import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import numpy as np
import base64
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
import matplotlib.pyplot as plt
import io

# ===============================
# CONFIGURACIÓN DE RUTA
# ===============================
DATASET_PATH = r"C:\Proyecto final\Dataset para PROYECTO FINAL\DEMALE-HSJM_2025_data.xlsx"
MODEL_PATH = "modelo_entrenado.pkl"

# Variables globales para el modelo
model = None
scaler = None
label_map = None
# Columnas base de síntomas y columnas opcionales de laboratorio para mejorar precisión
base_cols = ['fever', 'rash', 'abdominal_pain', 'dizziness', 'chills']
optional_cols = ['hemoglobin', 'red_blood_cells', 'white_blood_cells']
# expected_cols se determinará dinámicamente al entrenar/cargar el modelo
expected_cols = base_cols + optional_cols

# ===============================
# CARGAR O ENTRENAR MODELO
# ===============================
def cargar_modelo():
    """Carga el modelo entrenado si existe, si no, lo entrena"""
    global model, scaler, label_map
    
    if os.path.exists(MODEL_PATH):
        print(f"📂 Cargando modelo existente desde: {MODEL_PATH}")
        modelo_data = joblib.load(MODEL_PATH)
        model = modelo_data["model"]
        scaler = modelo_data["scaler"]
        label_map = modelo_data["label_map"]
        print("✅ Modelo cargado correctamente.")
    else:
        print("⚠️ No existe modelo entrenado. Entrenando nuevo modelo...")
        entrenar_modelo()
        
def entrenar_modelo():
    """Entrena el modelo desde cero"""
    global model, scaler, label_map
    
    print(f"📂 Cargando dataset desde: {DATASET_PATH}")
    
    try:
        data = pd.read_excel(DATASET_PATH)
    except FileNotFoundError:
        raise FileNotFoundError("❌ No se encontró el archivo Excel en la ruta indicada.")
    except Exception as e:
        raise RuntimeError(f"Error al leer el dataset: {e}")
    
    print("✅ Dataset cargado correctamente.")
    print(f"🔢 Filas: {data.shape[0]}, Columnas: {data.shape[1]}")
    
    # Normalizar nombres de columnas
    data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
    
    # Verificar columna objetivo
    target_col = 'disease'
    if target_col not in data.columns:
        print("⚠️ No se encontró la columna 'disease'. Buscando columnas similares...")
        posibles = [c for c in data.columns if 'enfer' in c or 'diag' in c or 'clas' in c or 'tipo' in c]
        if posibles:
            target_col = posibles[0]
            print(f"🧠 Usando columna '{target_col}' como variable objetivo.")
        else:
            raise ValueError("❌ No se encontró ninguna columna que indique la enfermedad (disease).")
    
    # Detectar qué columnas de entrada usaremos (aceptando opcionales si existen)
    cols_presentes = [c for c in base_cols + optional_cols if c in data.columns]
    # Asegurar que las columnas base estén presentes
    faltan_base = [c for c in base_cols if c not in cols_presentes]
    if faltan_base:
        raise ValueError(f"❌ Faltan columnas base requeridas en el dataset: {faltan_base}")

    # Establecer expected_cols global según dataset disponible
    global expected_cols
    expected_cols = cols_presentes

    # Separar variables y etiquetas
    X = data[expected_cols]
    y = data[target_col]
    
    # Mapear valores
    label_map = {
        1: 'Dengue',
        2: 'Malaria',
        3: 'Leptospirosis'
    }
    
    y = y.map(lambda v: label_map.get(v, v))
    
    print("🧬 Distribución de clases:")
    print(y.value_counts())
    
    # Entrenar modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    accuracy = model.score(X_test_scaled, y_test)
    print(f"✅ Modelo entrenado correctamente con una precisión del {accuracy*100:.2f}%")
    
    # Guardar modelo
    joblib.dump({"model": model, "scaler": scaler, "label_map": label_map}, MODEL_PATH)
    print(f"💾 Modelo guardado en: {os.path.abspath(MODEL_PATH)}")

# ===============================
# FUNCIÓN DE PREDICCIÓN
# ===============================
def predecir_enfermedad(sintomas):
    """
    Realiza la predicción de enfermedad.
    Acepta 5 síntomas binarios base y opcionalmente variables de laboratorio:
    - Base: fever, rash, abdominal_pain, dizziness, chills (0/1)
    - Opcionales: hemoglobin, red_blood_cells, white_blood_cells (numéricas)
    """
    if model is None or scaler is None:
        cargar_modelo()

    # Construir entrada alineada a expected_cols actual
    fila = []
    if isinstance(sintomas, dict):
        for col in expected_cols:
            if col in ['fever', 'rash', 'abdominal_pain', 'dizziness', 'chills']:
                valor = int(sintomas.get(col, 0))
            else:
                v = sintomas.get(col, None)
                # Permitir None para opcionales: imputar con 0 si no viene
                if v is None or v == "":
                    valor = 0.0
                else:
                    valor = float(v)
            fila.append(valor)
    else:
        # Compatibilidad: si viene como tupla/lista en orden base, y opcionales faltan
        base_vals = list(sintomas[:5])
        base_vals = [int(x) for x in base_vals]
        opc_map = {k: 0.0 for k in optional_cols}
        # Si trae más valores, mapear en orden opcional
        if len(sintomas) > 5:
            extra = sintomas[5:]
            for i, k in enumerate(optional_cols[:len(extra)]):
                opc_map[k] = float(extra[i])
        valores = {**dict(zip(base_cols, base_vals)), **opc_map}
        for col in expected_cols:
            fila.append(valores.get(col, 0.0))

    entrada = pd.DataFrame([fila], columns=expected_cols)
    entrada_scaled = scaler.transform(entrada)
    pred = model.predict(entrada_scaled)[0]
    return pred

# ===============================
# FUNCIONES DE EVALUACIÓN
# ===============================
def evaluar_modelo():
    """Evalúa el modelo y genera métricas con gráficos"""
    if model is None or scaler is None:
        cargar_modelo()
    
    print(f"📂 Cargando dataset para evaluación desde: {DATASET_PATH}")
    
    try:
        data = pd.read_excel(DATASET_PATH)
    except Exception as e:
        raise RuntimeError(f"Error al leer el dataset: {e}")
    
    # Normalizar nombres de columnas
    data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
    
    # Obtener datos
    target_col = 'disease'
    if target_col not in data.columns:
        posibles = [c for c in data.columns if 'enfer' in c or 'diag' in c or 'clas' in c or 'tipo' in c]
        if posibles:
            target_col = posibles[0]
    
    X = data[expected_cols]
    y_true = data[target_col].map(lambda v: label_map.get(v, v))
    
    # Predecir con el modelo
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    
    # Generar gráficos
    fig_confusion = generar_matriz_confusion(y_true, y_pred)
    fig_dispersion = generar_grafica_dispersion(X_scaled, y_true, y_pred)
    
    # Convertir gráficos a base64
    confusion_b64 = figura_a_base64(fig_confusion)
    dispersion_b64 = figura_a_base64(fig_dispersion)
    
    plt.close('all')  # Cerrar todas las figuras
    
    return {
        'precision': accuracy,
        'matriz_confusion': confusion_matrix(y_true, y_pred).tolist(),
        'reporte': classification_report(y_true, y_pred, output_dict=True),
        'grafica_confusion': confusion_b64,
        'grafica_dispersion': dispersion_b64
    }

def generar_matriz_confusion(y_true, y_pred):
    """Genera gráfica de matriz de confusión"""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Etiquetas
    clases = list(label_map.values())
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=clases, yticklabels=clases,
           title='Matriz de Confusión',
           ylabel='Etiqueta Real',
           xlabel='Etiqueta Predicha')
    
    # Agregar valores en cada celda
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    return fig

def generar_grafica_dispersion(X_scaled, y_true, y_pred):
    """Genera gráfica de dispersión usando PCA para reducir dimensiones"""
    from sklearn.decomposition import PCA
    
    # Si y_true es None, solo mostramos predicciones
    if y_true is None:
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Reducir a 2D con PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    
    # Mapeo de colores
    colores = {'Dengue': '#fbbf24', 'Malaria': '#10b981', 'Leptospirosis': '#3b82f6'}
    
    # Subplot 1: Etiquetas reales (solo si y_true existe)
    if y_true is not None:
        for enfermedad in label_map.values():
            mask = y_true == enfermedad
            ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colores.get(enfermedad, 'gray'),
                       label=enfermedad, alpha=0.6, s=50)
        ax1.set_title('Distribución Real de Enfermedades')
        ax1.set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax1.set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Predicciones del modelo
    for enfermedad in label_map.values():
        mask = y_pred == enfermedad
        ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colores.get(enfermedad, 'gray'),
                   label=enfermedad, alpha=0.6, s=50)
    ax2.set_title('Predicciones del Modelo')
    ax2.set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def figura_a_base64(fig):
    """Convierte una figura de matplotlib a base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

# ===============================
# FUNCIONES DE PREDICCIÓN POR LOTES
# ===============================
def predecir_lote(archivo_excel):
    """
    Predice enfermedades para un archivo Excel cargado.
    Retorna predicciones, gráficas y métricas.
    """
    if model is None or scaler is None:
        cargar_modelo()
    
    try:
        # Leer el archivo Excel
        data = pd.read_excel(archivo_excel)
    except Exception as e:
        raise RuntimeError(f"Error al leer el archivo Excel: {e}")
    
    # Normalizar nombres de columnas
    data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
    
    # Detectar columnas disponibles en el archivo (base + opcionales)
    cols_presentes = [c for c in base_cols + optional_cols if c in data.columns]
    faltan_base = [c for c in base_cols if c not in cols_presentes]
    if faltan_base:
        raise ValueError(f"Faltan columnas base requeridas en el archivo: {faltan_base}")

    # Alinear columnas usadas a las del modelo si es posible; si el modelo espera
    # más columnas que el archivo, imputar faltantes con 0.
    cols_para_usar = expected_cols if all(c in data.columns for c in expected_cols) else cols_presentes

    # Preparar datos alineados al modelo
    X = data.reindex(columns=expected_cols, fill_value=0.0)
    
    # Predecir
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    # Obtener probabilidades
    y_proba = model.predict_proba(X_scaled)
    
    # Agregar predicciones al DataFrame
    data['prediccion'] = y_pred
    for idx, enfermedad in enumerate(label_map.values()):
        data[f'prob_{enfermedad.lower()}'] = y_proba[:, idx]
    
    # Calcular estadísticas
    estadisticas = {
        'total_casos': len(data),
        'distribucion': pd.Series(y_pred).value_counts().to_dict()
    }
    
    # Generar gráficas
    fig_confusion = None
    fig_dispersion = generar_grafica_dispersion(X_scaled, None, y_pred)
    
    # Si hay columna 'disease' o similar, calcular matriz de confusión
    target_col = None
    for col in data.columns:
        if 'disease' in col or 'enfer' in col or 'diag' in col:
            target_col = col
            break
    
    if target_col:
        try:
            # Intentar mapear los valores reales
            y_true = data[target_col].map(lambda v: label_map.get(v, v))
            fig_confusion = generar_matriz_confusion(y_true, y_pred)
            confusion_b64 = figura_a_base64(fig_confusion)
            estadisticas['precision'] = accuracy_score(y_true, y_pred)
        except:
            confusion_b64 = None
            estadisticas['precision'] = None
    else:
        confusion_b64 = None
        estadisticas['precision'] = None
    
    dispersion_b64 = figura_a_base64(fig_dispersion)
    
    plt.close('all')
    
    # Convertir DataFrame a formato JSON
    resultados_df = data.to_dict('records')
    
    return {
        'predicciones': resultados_df,
        'estadisticas': estadisticas,
        'grafica_confusion': confusion_b64,
        'grafica_dispersion': dispersion_b64
    }

# ===============================
# INICIALIZACIÓN
# ===============================
# Cargar modelo al importar el módulo
cargar_modelo()

# ===============================
# TEST LOCAL
# ===============================
if __name__ == "__main__":
    print("\n🧠 Test de ejemplo:")
    ejemplo = predecir_enfermedad({'fever': 1, 'rash': 0, 'abdominal_pain': 1, 'dizziness': 0, 'chills': 1})
    print(f"Predicción para fiebre=1, rash=0, dolor_abd=1, mareos=0, escalofríos=1 → {ejemplo}")
    
    print("\n📊 Evaluando modelo...")
    resultados = evaluar_modelo()
    print(f"Precisión: {resultados['precision']*100:.2f}%")
