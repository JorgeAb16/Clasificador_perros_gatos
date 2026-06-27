import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Clasificador Perros y Gatos IA", layout="centered")
st.title("Modelo predictivo Perros y Gatos - Clase de IA")
st.write("Suba una imagen para clasificar con el Modelo MobileNetV2 pre-entrenado Jorge Abraham Fajardo 20231900189")

IMG_SIZE = (224, 224)
MODEL_PATHS = [
    Path("modelo_perros_gatos.keras"),
    Path("modelo_perros_gatos.h5"),
]
CLASS_PATH = Path("clases.json")

# --- DICCIONARIO DE TRADUCCIÓN ---
LABELS_ES = {
    "Gatos": "Gato",
    "Perros": "Perro",
}

# --- FUNCIONES DE CARGA ---
@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)
    st.error("No se encontró el modelo. Asegúrese de que modelo_perros_gatos.h5 esté junto a app.py.")
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
    top2 = np.argsort(preds)[-2:][::-1]
    return [
        (LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in top2
    ]

modelo = cargar_modelo()
clases = cargar_clases()

archivo = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen analizada", use_container_width=True)

    resultados = predecir(imagen)
    st.subheader("Resultado")
    st.success(f"Predicción : {resultados[0][0]}")
else:
    st.info("Cargue una imagen para iniciar la clasificación.")
