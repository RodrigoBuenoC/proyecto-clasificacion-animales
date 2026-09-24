import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

# Cargar el modelo optimizado para que no consuma toda la RAM [cite: 56, 61]
session = ort.InferenceSession("modelo_optimizado.onnx")
# Etiquetas en el orden de tu entrenamiento (0: Elefante, 1: Gallina, 2: Perro)
etiquetas = ["Elefante", "Gallina", "Perro"]

def predecir(imagen_pil):
    # Ajustar imagen a 150x150 como en el entrenamiento de Colab
    imagen_pil = imagen_pil.resize((150, 150))
    img_array = np.array(imagen_pil).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Inferencia ultraligera [cite: 66, 67]
    input_name = session.get_inputs()[0].name
    prediccion = session.run(None, {input_name: img_array})[0]
    
    # Devolver probabilidades para la interfaz de Gradio [cite: 68]
    return {etiquetas[i]: float(prediccion[0][i]) for i in range(3)}

# Crear la interfaz visual [cite: 69, 70]
demo = gr.Interface(
    fn=predecir,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Clasificador de Animales"
)

demo.launch()