# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
from modelo import predecir_enfermedad, evaluar_modelo, predecir_lote, cargar_modelo
import json
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Crear carpeta de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==============================
# RUTA PRINCIPAL
# ==============================
@app.route('/')
def home():
    # Flask lee los parámetros del modelo y los inyecta en la plantilla.
    # Esto es necesario para que el frontend pueda acceder a ellos si es necesario,
    # aunque la predicción principal se haga en el servidor.
    try:
        with open('model_params.json', 'r') as f:
            model_params_json = f.read()
    except FileNotFoundError:
        model_params_json = "{}"  # Enviar un objeto JSON vacío si no se encuentra
    return render_template('index.html', model_params=model_params_json)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# ==============================
# RUTA PARA SERVIR EL MODELO JSON
# ==============================
@app.route('/model_params.json')
def send_model_params():
    return send_from_directory('.', 'model_params.json')

# ==============================
# PREDICCIÓN DESDE EL FORMULARIO
# ==============================
@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # Recoger todos los datos del formulario dinámicamente
        sintomas = request.form.to_dict()
        # Extraer el tipo de modelo, con un valor por defecto si no se envía
        tipo_modelo = sintomas.pop('model_type', 'logistica')
        resultado = predecir_enfermedad(sintomas, tipo_modelo=tipo_modelo)
        return jsonify({'resultado': resultado})
    except Exception as e:
        return jsonify({'error': str(e)})

# ==============================
# EVALUACIÓN DEL MODELO
# ==============================
@app.route('/evaluar', methods=['GET'])
def evaluar():
    try:
        resultados = evaluar_modelo()
        return jsonify(resultados)
    except Exception as e:
        return jsonify({'error': str(e)})

# ==============================
# PREDICCIÓN POR LOTES (CARGA EXCEL)
# ==============================
@app.route('/predecir_lote', methods=['POST'])
def predecir_lote_endpoint():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400
        
        file = request.files['file'] 
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        # Verificar extensión
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            return jsonify({'error': 'El archivo debe ser de tipo Excel (.xlsx, .xls) o CSV (.csv)'}), 400
        
        # Guardar archivo temporalmente
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Procesar con el modelo
        resultados = predecir_lote(filepath)
        
        # Eliminar archivo temporal
        os.remove(filepath)
        
        return jsonify(resultados)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        # Limpiar archivo si existe
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error al procesar archivo: {str(e)}'}), 500

# ==============================
# EJECUCIÓN LOCAL
# ==============================
if __name__ == '__main__':
    cargar_modelo() # Carga el modelo al iniciar la app
    app.run(debug=True)
