# crear_standalone.py
import json
import os

# --- Configuración ---
TEMPLATE_PATH = os.path.join("templates", "index.html")
MODEL_PARAMS_PATH = "model_params.json"
OUTPUT_PATH = "app_standalone.html"  # El archivo final y autónomo

print("🚀 Creando aplicación autónoma...")

# 1. Leer los parámetros del modelo
try:
    with open(MODEL_PARAMS_PATH, 'r', encoding='utf-8') as f:
        model_params_json = f.read()
    print(f"✅ Parámetros del modelo leídos desde '{MODEL_PARAMS_PATH}'.")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo '{MODEL_PARAMS_PATH}'.")
    print("👉 Por favor, ejecuta 'python export_model.py' primero para generar este archivo.")
    exit()

# 2. Leer la plantilla HTML
try:
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        html_content = f.read()
    print(f"✅ Plantilla HTML leída desde '{TEMPLATE_PATH}'.")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo de plantilla '{TEMPLATE_PATH}'.")
    exit()

# 3. Reemplazar el marcador de posición con los datos JSON reales
placeholder = "{{ model_params | safe }}"
final_html = html_content.replace(placeholder, model_params_json)
print("🧩 Inyectando parámetros del modelo en el HTML...")

# 4. Guardar el archivo HTML final y autónomo
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(final_html)

print(f"\n🎉 ¡Éxito! Tu aplicación autónoma ha sido creada en: '{os.path.abspath(OUTPUT_PATH)}'")
print("👉 Ahora puedes abrir ese archivo directamente en tu navegador.")