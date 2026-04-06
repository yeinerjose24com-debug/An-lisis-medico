# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from flask_wtf.csrf import CSRFProtect
from modelo import predecir_enfermedad, evaluar_modelo, predecir_lote, cargar_modelo
import json
import os
import socket

app = Flask(__name__)
# Secret key required for session support; in production use a secure value
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')
csrf = CSRFProtect(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Crear carpeta de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.after_request
def add_security_headers(response):
    """
    Agrega cabeceras de seguridad robustas, incluyendo protección contra Clickjacking 
    y una Content Security Policy (CSP) completa con directiva de respaldo.
    """
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Definición de CSP con default-src para evitar el error de "No Fallback"
    # Se habilitan los recursos necesarios: scripts de CDN, estilos inline y fuentes de Google.
    csp_policy = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "frame-ancestors 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    response.headers['Content-Security-Policy'] = csp_policy
    return response

# ==============================
# RUTA DE LOGIN / LOGOUT
# ==============================
@app.route('/login', methods=['GET', 'POST'])
def login():
  
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        # debug output (appears in server console)
        app.logger.debug(f"Intento de inicio de sesión para el usuario: {username!r}")

        # simple check; change as desired
        if username == 'yeinner' and password == 'admin':
            session['user'] = username
            session.permanent = False
            app.logger.debug("Usuario es correcto , redirije a pagina inicial ")
            return redirect(url_for('home'))
        else:
            msg = 'Credenciales inválidas. Por favor, intente de nuevo.'
            app.logger.debug(f"Login failed: {msg}")
            return render_template('login.html', error=msg)
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ==============================
# RUTA PRINCIPAL
# ==============================
@app.route('/')
def home():
    # proteger acceso a la página principal
    if 'user' not in session:
        return redirect(url_for('login'))

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
    def get_local_ip():
        try:
            # Crea un socket temporal para obtener la IP local real
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except:
            return "127.0.0.1"

    # Detectar si estamos en producción (ej. Render) o local
    # Las plataformas en la nube siempre definen la variable de entorno 'PORT'
    is_prod = "PORT" in os.environ
    
    port = int(os.environ.get("PORT", 5000))
    # Usamos '0.0.0.0' para permitir que el celular y otros equipos en la misma Wi-Fi entren.
    # Si en Windows parece que se queda trabado al inicio, espera 20 segundos, es normal.
    host = "0.0.0.0"
    # En local, desactivar el reloader ayuda a evitar que Windows bloquee el puerto por intentos duplicados
    debug_mode = not is_prod
    local_ip = get_local_ip()
    
    # Cargamos el modelo si no estamos en modo debug o si somos el proceso hijo del reloader.
    if not debug_mode or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print(f"\n" + "="*50)
        print(f"🚀 SERVIDOR INICIANDO...")
        print(f"📢 LOCAL:  http://127.0.0.1:{port}")
        print(f"🌐 WI-FI:  http://{local_ip}:{port}")
        print(f"="*50)
        print(f"⚠️  IMPORTANTE: Si el celular no conecta, verifica que el Firewall")
        print(f"   de Windows permita el puerto {port}.\n")
        print("🧠 Cargando IA... Por favor espera a que diga 'LISTO'.")
        try:
            cargar_modelo()
            print("✅ ¡LISTO! El servidor ya acepta conexiones.\n")
        except Exception as e:
            print(f"❌ Error crítico al cargar el modelo: {e}")

    app.run(host=host, port=port, debug=debug_mode, use_reloader=debug_mode)
