import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
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
# Ruta relativa para compatibilidad con Render y ejecución local.
DATASET_PATH = "data/DEMALE-HSJM_2025_data.xlsx"
MODEL_PATH = "modelo_entrenado.pkl"

# Variables globales para el modelo
model_logistica = None
model_mlp = None
scaler = None
label_map = None
# Lista completa de TODAS las variables predictoras posibles
expected_cols = [
    'male', 'female', 'age', 'urban_origin', 'rural_origin',
    'homemaker', 'student', 'professional', 'merchant', 'agriculture_livestock',
    'various_jobs', 'unemployed', 'hospitalization_days', 'body_temperature',
    'fever', 'headache', 'dizziness', 'loss_of_appetite', 'weakness',
    'myalgias', 'arthralgias', 'eye_pain', 'hemorrhages', 'vomiting',
    'abdominal_pain', 'chills', 'hemoptysis', 'edema', 'jaundice',
    'bruises', 'petechiae', 'rash', 'diarrhea', 'respiratory_difficulty',
    'itching', 'hematocrit', 'hemoglobin', 'red_blood_cells',
    'white_blood_cells', 'neutrophils', 'eosinophils', 'basophils',
    'monocytes', 'lymphocytes', 'platelets', 'ast', 'alt',
    'alp', 'total_bilirubin', 'direct_bilirubin',
    'indirect_bilirubin', 'total_proteins', 'albumin', 'creatinine', 'urea'
]

# Renombrar columnas para que coincidan con el dataset
column_rename_map = {
    'sgot': 'ast',
    'sgpt': 'alt',
    'alkaline_phosphatase': 'alp'
}

# ===============================
# VALIDACIÓN DE DATOS
# ===============================
def validar_sintomas(sintomas):
    """Valida que los datos de entrada estén en rangos realistas."""
    errores = []
    age = sintomas.get('age')
    if age is not None and not (0 < float(age) < 120):
        errores.append("La edad debe ser un valor realista (entre 1 y 119).")
    
    temp = sintomas.get('body_temperature')
    if temp is not None and not (34 < float(temp) < 45):
        errores.append("La temperatura corporal debe estar entre 34°C y 45°C.")

    hemoglobin = sintomas.get('hemoglobin')
    if hemoglobin is not None and not (5 < float(hemoglobin) < 25):
        errores.append("El valor de hemoglobina es irreal (debe estar entre 5 y 25 g/dL).")

    if errores:
        raise ValueError("Datos no válidos: " + " ".join(errores))
# ===============================
# CARGAR O ENTRENAR MODELO
# ===============================
def cargar_modelo():
    """Carga el modelo entrenado si existe, si no, lo entrena"""
    global model_logistica, model_mlp, scaler, label_map, expected_cols
    
    if os.path.exists(MODEL_PATH):
        print(f"📂 Cargando modelo existente desde: {MODEL_PATH}")
        modelo_data = joblib.load(MODEL_PATH)
        
        # Comprobar si el modelo cargado es compatible (tiene los dos modelos)
        if "model_logistica" in modelo_data and "model_mlp" in modelo_data:
            model_logistica = modelo_data["model_logistica"]
            model_mlp = modelo_data["model_mlp"]
            scaler = modelo_data["scaler"]
            expected_cols = modelo_data["columns"]
            label_map = modelo_data["label_map"]
            print("✅ Modelo cargado correctamente.")
        else:
            # Si es un modelo antiguo, forzar re-entrenamiento
            print("⚠️ Modelo antiguo detectado. Forzando re-entrenamiento para actualizarlo...")
            entrenar_modelo()
    else:
        print("⚠️ No existe modelo entrenado. Entrenando nuevo modelo...")
        entrenar_modelo()
        
def entrenar_modelo():
    """Entrena el modelo desde cero"""
    global model_logistica, model_mlp, scaler, label_map, expected_cols
    
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
    data.columns = [c.strip().lower().replace(" ", "_").replace("(sgot)", "").replace("(sgpt)", "") for c in data.columns]
    data.rename(columns=column_rename_map, inplace=True)
    
    # Verificar columna objetivo
    target_col_name = 'diagnosis' # Cambiado a 'diagnosis'
    if target_col_name not in data.columns:
        print("⚠️ No se encontró la columna 'disease'. Buscando columnas similares...")
        posibles = [c for c in data.columns if 'enfer' in c or 'diag' in c or 'clas' in c or 'tipo' in c]
        if posibles:
            target_col_name = posibles[0]
            print(f"🧠 Usando columna '{target_col_name}' como variable objetivo.")
        else:
            raise ValueError("❌ No se encontró ninguna columna que indique la enfermedad (disease).")
    
    # Detectar qué columnas de entrada usaremos (aceptando opcionales si existen)
    cols_a_usar = [col for col in globals()['expected_cols'] if col in data.columns]
    if not cols_a_usar:
        raise ValueError("❌ Ninguna de las columnas esperadas se encontró en el dataset.")

    # Separar variables y etiquetas
    X = data[cols_a_usar]
    y = data[target_col_name]
    
    # Mapear valores
    label_map = {
        1: 'Dengue',
        2: 'Malaria',
        3: 'Leptospirosis'
    }
    
    y = y.map(lambda v: label_map.get(v, v))
    
    print("🧬 Distribución de clases original:")
    print(y.value_counts())
    
    # --- NUEVO: Forzar un balanceo perfecto a 50 muestras por clase ---
    print("\n⚖️ Forzando balanceo a 50 muestras por clase...")
    N_SAMPLES_PER_CLASS = 51
    X_balanced_list = []
    y_balanced_list = []

    for disease_name in label_map.values():
        # Filtrar datos por clase
        X_class = X[y == disease_name]
        y_class = y[y == disease_name]
        
        current_samples = len(X_class)
        print(f"   - Clase '{disease_name}': {current_samples} muestras encontradas.")

        if current_samples > N_SAMPLES_PER_CLASS:
            # Submuestreo (undersampling) si hay más de 50
            print(f"     -> Reduciendo a {N_SAMPLES_PER_CLASS} muestras.")
            X_res, y_res = X_class.sample(n=N_SAMPLES_PER_CLASS, random_state=42), y_class.sample(n=N_SAMPLES_PER_CLASS, random_state=42)
        elif current_samples < N_SAMPLES_PER_CLASS:
            # Sobremuestreo (oversampling) si hay menos de 50
            print(f"     -> Aumentando a {N_SAMPLES_PER_CLASS} muestras (con reemplazo).")
            X_res, y_res = X_class.sample(n=N_SAMPLES_PER_CLASS, replace=True, random_state=42), y_class.sample(n=N_SAMPLES_PER_CLASS, replace=True, random_state=42)
        else:
            # Si ya tiene 50, se usa tal cual
            X_res, y_res = X_class, y_class

        X_balanced_list.append(X_res)
        y_balanced_list.append(y_res)

    # Combinar los dataframes balanceados
    X = pd.concat(X_balanced_list)
    y = pd.concat(y_balanced_list)

    print("\n🧬 Distribución de clases después del balanceo forzado:")
    print(y.value_counts())
    
    # Dividir los datos YA BALANCEADOS en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar los modelos con los datos balanceados y escalados.
    model_logistica = LogisticRegression(random_state=42, max_iter=1000)
    model_logistica.fit(X_train_scaled, y_train)
    accuracy_logistica = model_logistica.score(X_test_scaled, y_test)
    print(f"✅ Modelo de Regresión Logística entrenado con una precisión del {accuracy_logistica*100:.2f}%")

    model_mlp = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50), alpha=0.0001, solver='adam', learning_rate='adaptive')
    model_mlp.fit(X_train_scaled, y_train)
    accuracy_mlp = model_mlp.score(X_test_scaled, y_test)
    print(f"✅ Modelo de Red Neuronal (MLP) entrenado con una precisión del {accuracy_mlp*100:.2f}%")
    
    # Guardar modelo
    expected_cols = cols_a_usar # Actualizar la variable global con las columnas usadas
    joblib.dump({
        "model_logistica": model_logistica, "model_mlp": model_mlp, 
        "scaler": scaler, "label_map": label_map, "columns": expected_cols
    }, MODEL_PATH)
    print(f"💾 Modelo guardado en: {os.path.abspath(MODEL_PATH)}")

# ===============================
# FUNCIÓN DE PREDICCIÓN
# ===============================
def predecir_enfermedad(sintomas, tipo_modelo='logistica'):
    """
    Realiza la predicción de enfermedad usando el conjunto completo de variables.
    """
    if model_logistica is None or model_mlp is None or scaler is None:
        cargar_modelo()

    # Validar datos de entrada
    validar_sintomas(sintomas)

    # Construir la fila de datos para la predicción
    fila = {}
    for col in expected_cols:
        valor = sintomas.get(col)
        if valor is None or valor == '':
            fila[col] = 0.0  # Imputar con 0 si no se proporciona
        else:
            fila[col] = float(valor)

    # Crear DataFrame y predecir
    entrada = pd.DataFrame([fila], columns=expected_cols)
    entrada_scaled = scaler.transform(entrada)

    # Seleccionar el modelo según la elección del usuario
    if tipo_modelo == 'red_neuronal':
        print("🧠 Usando modelo de predicción: Red Neuronal (MLP)")
        pred = model_mlp.predict(entrada_scaled)[0]
    else:
        print("🧠 Usando modelo de predicción: Regresión Logística")
        pred = model_logistica.predict(entrada_scaled)[0]
        
        
    return pred

# ===============================
# FUNCIONES DE EVALUACIÓN
# ===============================
def evaluar_modelo():
    """Evalúa ambos modelos y genera métricas con gráficos"""
    if model_logistica is None or model_mlp is None or scaler is None:
        cargar_modelo()
    
    print(f"📂 Cargando dataset para evaluación desde: {DATASET_PATH}")
    
    try:
        data = pd.read_excel(DATASET_PATH)
    except Exception as e:
        raise RuntimeError(f"Error al leer el dataset: {e}")
    
    # Normalizar nombres de columnas
    data.columns = [c.strip().lower().replace(" ", "_").replace("(sgot)", "").replace("(sgpt)", "") for c in data.columns]
    data.rename(columns=column_rename_map, inplace=True)
    
    # Obtener datos
    target_col_name = 'diagnosis'
    if target_col_name not in data.columns:
        posibles = [c for c in data.columns if 'enfer' in c or 'diag' in c or 'clas' in c or 'tipo' in c]
        if posibles:
            target_col_name = posibles[0]
    
    # --- NUEVO: Forzar balanceo a 50 muestras por clase para la evaluación ---
    # Esto asegura que la matriz de confusión refleje un dataset balanceado.
    print("\n⚖️ Forzando balanceo a 50 muestras por clase para evaluación...")
    N_SAMPLES_PER_CLASS = 51
    
    # Combinar X e y temporalmente para el muestreo
    X_original = data.reindex(columns=expected_cols, fill_value=0.0)
    y_original = data[target_col_name].map(lambda v: label_map.get(v, v))
    
    data_for_eval = X_original.copy()
    data_for_eval['__target__'] = y_original
    
    balanced_dfs = []
    for disease_name in label_map.values():
        class_df = data_for_eval[data_for_eval['__target__'] == disease_name]
        current_samples = len(class_df)
        
        if current_samples > N_SAMPLES_PER_CLASS:
            res_df = class_df.sample(n=N_SAMPLES_PER_CLASS, random_state=42)
        elif current_samples < N_SAMPLES_PER_CLASS:
            res_df = class_df.sample(n=N_SAMPLES_PER_CLASS, replace=True, random_state=42)
        else:
            res_df = class_df
        balanced_dfs.append(res_df)
        
    balanced_data = pd.concat(balanced_dfs)
    
    # Separar X e y del dataset ya balanceado
    X = balanced_data.drop(columns=['__target__'])
    y_true = balanced_data['__target__']
    
    print("\n🧬 Distribución de clases para evaluación:")
    print(y_true.value_counts())
    
    # Predecir con el modelo
    X_scaled = scaler.transform(X)
    y_pred_logistica = model_logistica.predict(X_scaled)
    y_pred_mlp = model_mlp.predict(X_scaled)
    
    # Calcular métricas para Regresión Logística
    accuracy_logistica = accuracy_score(y_true, y_pred_logistica)
    
    # Generar gráficos para Regresión Logística
    fig_confusion_logistica = generar_matriz_confusion(y_true, y_pred_logistica, "Regresión Logística")
    fig_dispersion_logistica = generar_grafica_dispersion(X_scaled, y_true, y_pred_logistica, "Regresión Logística")
    
    # Convertir gráficos a base64
    confusion_b64_logistica = figura_a_base64(fig_confusion_logistica)
    dispersion_b64_logistica = figura_a_base64(fig_dispersion_logistica)

    # Calcular métricas para Red Neuronal
    accuracy_mlp = accuracy_score(y_true, y_pred_mlp)
    
    # Generar gráficos para Red Neuronal
    fig_confusion_mlp = generar_matriz_confusion(y_true, y_pred_mlp, "Red Neuronal (MLP)")
    fig_dispersion_mlp = generar_grafica_dispersion(X_scaled, y_true, y_pred_mlp, "Red Neuronal (MLP)")
    
    # Convertir gráficos a base64
    confusion_b64_mlp = figura_a_base64(fig_confusion_mlp)
    dispersion_b64_mlp = figura_a_base64(fig_dispersion_mlp)
    
    plt.close('all')  # Cerrar todas las figuras
    
    return {
        'logistica': {
            'precision': accuracy_logistica,
            'grafica_confusion': confusion_b64_logistica,
            'grafica_dispersion': dispersion_b64_logistica
        },
        'red_neuronal': {
            'precision': accuracy_mlp,
            'grafica_confusion': confusion_b64_mlp,
            'grafica_dispersion': dispersion_b64_mlp
        }
    }

def generar_matriz_confusion(y_true, y_pred, model_name=""):
    """Genera gráfica de matriz de confusión"""
    fig, ax = plt.subplots(figsize=(8, 6))
    clases = list(label_map.values())
    cm = confusion_matrix(y_true, y_pred, labels=clases)
    
    # Usar un colormap más moderno y agradable
    im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
    
    # Añadir una barra de color para referencia
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Número de casos', rotation=-90, va="bottom")
    
    # Etiquetas
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=clases, yticklabels=clases,
           title=f'Matriz de Confusión ({model_name})',
           ylabel='Etiqueta Real',
           xlabel='Etiqueta Predicha')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Agregar valores en cada celda
    # El umbral de color de texto se ajusta para el colormap 'viridis'
    thresh = cm.max() * 0.55
    row_sums = cm.sum(axis=1) # Suma de cada fila (total de casos reales por clase)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            # Evitar división por cero si una clase no tiene muestras
            percentage = f"{(count / row_sums[i] * 100):.1f}%" if row_sums[i] > 0 else "0.0%"
            ax.text(j, i, f"{count}\n({percentage})",
                   ha="center", va="center",
                   color="white" if count > thresh else "black",
                   fontsize=10, weight='bold', linespacing=1.5)
    
    plt.tight_layout()
    return fig

def generar_grafica_dispersion(X_scaled, y_true, y_pred, model_name=""):
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
        ax1.set_xlabel(f'CP 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax1.set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Predicciones del modelo
    for enfermedad in label_map.values():
        mask = y_pred == enfermedad
        ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colores.get(enfermedad, 'gray'),
                   label=enfermedad, alpha=0.6, s=50)
    ax2.set_title(f'Predicciones ({model_name})')
    ax2.set_xlabel(f'CP 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'CP 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
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
    if model_logistica is None or scaler is None:
        cargar_modelo()
    
    try:
        # Leer el archivo Excel o CSV
        if archivo_excel.endswith('.csv'):
            data = pd.read_csv(archivo_excel)
        else:
            data = pd.read_excel(archivo_excel)

    except Exception as e:
        raise RuntimeError(f"Error al leer el archivo Excel: {e}")
    
    # Normalizar nombres de columnas
    data.columns = [c.strip().lower().replace(" ", "_").replace("(sgot)", "").replace("(sgpt)", "") for c in data.columns]
    data.rename(columns=column_rename_map, inplace=True)

    # Validar que al menos algunas columnas esperadas estén presentes
    if not any(col in data.columns for col in expected_cols):
        raise ValueError("El archivo no contiene ninguna de las columnas predictoras esperadas.")
    # Preparar datos alineados al modelo
    X = data.reindex(columns=expected_cols, fill_value=0.0)
    
    # Predecir
    # Para lotes, usamos el modelo más rápido (Regresión Logística) por defecto.
    X_scaled = scaler.transform(X)
    y_pred = model_logistica.predict(X_scaled)
    y_proba = model_logistica.predict_proba(X_scaled)
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
    fig_dispersion = generar_grafica_dispersion(X_scaled, None, y_pred, "Regresión Logística")
    
    # Si hay columna 'disease' o similar, calcular matriz de confusión
    target_col = None
    for col in data.columns:
        if 'diagnosis' in col or 'disease' in col or 'enfer' in col:
            target_col = col
            break
    
    if target_col:
        # Asegurarse de que la columna de verdad fundamental (ground truth) exista y no esté vacía
        if target_col in data and not data[target_col].isnull().all():
            # Mapear los valores reales y eliminar filas donde el mapeo falló (NaN)
            y_true_mapped = data[target_col].map(lambda v: label_map.get(v, v))
            
            # Crear un DataFrame temporal para alinear predicciones y valores reales
            # Esto es crucial si hay valores NaN en y_true_mapped que deben ser ignorados
            temp_df = pd.DataFrame({'true': y_true_mapped, 'pred': y_pred}).dropna()
            
            y_true = temp_df['true']
            y_pred_aligned = temp_df['pred']
            
            fig_confusion = generar_matriz_confusion(y_true, y_pred, "Regresión Logística")
            confusion_b64 = figura_a_base64(fig_confusion)
            estadisticas['precision'] = accuracy_score(y_true, y_pred_aligned)
        else:
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
    sintomas_ejemplo = {'age': 35, 'body_temperature': 39.5, 'fever': 1, 'rash': 0, 'abdominal_pain': 1, 'dizziness': 0, 'chills': 1}
    pred_logistica = predecir_enfermedad(sintomas_ejemplo, tipo_modelo='logistica')
    print(f"Predicción con Regresión Logística → {pred_logistica}")
    pred_mlp = predecir_enfermedad(sintomas_ejemplo, tipo_modelo='red_neuronal')
    print(f"Predicción con Red Neuronal → {pred_mlp}")
    
    print("\n📊 Evaluando modelo...")
    resultados = evaluar_modelo()
    print(f"Precisión (Logística): {resultados['logistica']['precision']*100:.2f}%")
    print(f"Precisión (Red Neuronal): {resultados['red_neuronal']['precision']*100:.2f}%")
