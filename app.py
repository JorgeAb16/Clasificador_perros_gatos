import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Clasificador Perros y Gatos",
    page_icon="🐾",
    layout="centered"
)

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}

.hero {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f7971e, #ffd200);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero p {
    color: #b0aed0;
    font-size: 0.95rem;
    margin-top: 0;
}

.badge {
    display: inline-block;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    color: #d4d0f0;
    margin-bottom: 1.5rem;
}

.upload-box {
    background: rgba(255,255,255,0.05);
    border: 2px dashed rgba(247,151,30,0.5);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
}

.result-card {
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    margin-top: 1.5rem;
}
.result-emoji {
    font-size: 4rem;
    margin-bottom: 0.3rem;
}
.result-label {
    font-size: 2rem;
    font-weight: 700;
    color: #ffd200;
}
.result-conf {
    font-size: 1rem;
    color: #b0aed0;
    margin-top: 0.2rem;
}

.bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #c0bde0;
    margin-bottom: 0.2rem;
    margin-top: 0.8rem;
}
.bar-track {
    background: rgba(255,255,255,0.1);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.bar-fill-perro {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #f7971e, #ffd200);
}
.bar-fill-gato {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #7f7fd5, #91eae4);
}

.footer {
    text-align: center;
    color: #5c5a7a;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-bottom: 1rem;
}

#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
MODEL_PATHS = [Path("modelo_perros_gatos.keras"), Path("modelo_perros_gatos.h5")]
CLASS_PATH  = Path("clases.json")

LABELS_ES = {"Gatos": "Gato", "Perros": "Perro"}
EMOJIS    = {"Gato": "🐱", "Perro": "🐶"}
BAR_CLASS = {"Gato": "bar-fill-gato", "Perro": "bar-fill-perro"}

# ── Carga de modelo ───────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)
    st.error("No se encontró el modelo. Coloca modelo_perros_gatos.keras junto a app.py.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["Gatos", "Perros"]

def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    return [
        (LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in np.argsort(preds)[::-1]
    ]

modelo = cargar_modelo()
clases = cargar_clases()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🐾 ¿Perro o Gato?</h1>
    <p>Clasificador de imágenes con Transfer Learning — MobileNetV2</p>
</div>
<div style="text-align:center">
    <span class="badge">Jorge Abraham Fajardo López · 20231900189</span>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
archivo = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if not archivo:
    st.markdown("""
    <div class="upload-box">
        <p style="font-size:2rem; margin:0">📂</p>
        <p style="color:#b0aed0; margin:0.3rem 0 0">Arrastra una imagen aquí o usa el botón de arriba</p>
        <p style="color:#5c5a7a; font-size:0.8rem; margin:0.2rem 0 0">Formatos: JPG · JPEG · PNG</p>
    </div>
    """, unsafe_allow_html=True)

else:
    imagen     = Image.open(archivo)
    resultados = predecir(imagen)
    top_label  = resultados[0][0]
    top_conf   = resultados[0][1]
    emoji      = EMOJIS.get(top_label, "🐾")

    st.image(imagen, caption="Imagen analizada", use_container_width=True)

    
    st.markdown("""
<div style="text-align:center; margin: 1.5rem 0 0.5rem;">
    <span style="
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #b0aed0;
    ">Resultado del análisis</span>
    <div style="
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        border-radius: 999px;
        margin: 0.4rem auto 0;
    "></div>
</div>
""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card">
         <div class="result-emoji">{emoji}</div>
         <div class="result-label">{top_label}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Clase de Inteligencia Artificial · Campus Comayagua 2026 · Powered by TensorFlow & MobileNetV2
</div>
""", unsafe_allow_html=True)

