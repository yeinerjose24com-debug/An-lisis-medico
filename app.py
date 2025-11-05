# app.py
from flask import Flask, render_template, request, jsonify
from modelo import predecir_enfermedad, evaluar_modelo, predecir_lote
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
    return render_template('index.html')

# ==============================
# PREDICCIÓN DESDE EL FORMULARIO
# ==============================
@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        sintomas = {
            'fever': int(request.form.get('fever', 0)),
            'rash': int(request.form.get('rash', 0)),
            'abdominal_pain': int(request.form.get('abdominal_pain', 0)),
            'dizziness': int(request.form.get('dizziness', 0)),
            'chills': int(request.form.get('chills', 0))
        }

        resultado = predecir_enfermedad(sintomas)
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
        if not file.filename.endswith(('.xlsx', '.xls')):
            return jsonify({'error': 'El archivo debe ser Excel (.xlsx o .xls)'}), 400
        
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
    app.run(debug=True)
