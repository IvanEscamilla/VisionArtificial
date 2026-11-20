"""Prueba del entorno · Práctica 4

Comprueba la versión de Python (TensorFlow necesita 3.10, 3.11 o 3.12), que las librerías están
instaladas, que TensorFlow puede convertir un modelo a TensorFlow Lite, que se leen las fotos de
recursos/ y que la webcam abre.
Ejecutar (con el entorno virtual activado):  python prueba_entorno.py
"""
import sys                        # información del intérprete de Python
from pathlib import Path          # rutas que funcionan en Windows, macOS y Linux

if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    sys.exit(f"Python {sys.version.split()[0]}: TensorFlow necesita Python 3.10, 3.11 o 3.12. "
             "Instala Python 3.12 y crea el entorno con:  py -3.12 -m venv va")

import cv2                        # OpenCV (paquete opencv-python)
import matplotlib                 # Matplotlib: gráficas y figuras
import numpy as np                # NumPy: arreglos numéricos
import tensorflow as tf           # TensorFlow (incluye Keras)

CARPETA = Path(__file__).resolve().parent

# 1) Versiones instaladas: si alguna importación falla, el script se detiene antes de llegar aquí
print("Python      :", sys.version.split()[0])
print("NumPy       :", np.__version__)
print("OpenCV      :", cv2.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("TensorFlow  :", tf.__version__)

# 2) Una red mínima: se crea, se convierte a TensorFlow Lite y se ejecuta con el intérprete
red = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(2)])
tflite = tf.lite.TFLiteConverter.from_keras_model(red).convert()
interprete = tf.lite.Interpreter(model_content=tflite)
interprete.allocate_tensors()
print(f"TFLite      : conversión e intérprete OK ({len(tflite)} bytes)")

# 3) Las fotos de recursos/ (para trabajar sin webcam con --fotos)
fotos = sorted((CARPETA / "recursos").glob("*.png"))
print(f"Fotos       : {len(fotos)} en recursos/" if fotos else "Fotos       : NO se encontró recursos/")

# 4) La webcam: se abre, se toma un frame y se libera
camara = cv2.VideoCapture(0)
ok_camara, frame = camara.read() if camara.isOpened() else (False, None)
camara.release()
if ok_camara:
    print(f"Webcam      : {frame.shape[1]}x{frame.shape[0]}")
else:
    print("Webcam      : no disponible (cierra Zoom/Teams o usa --fotos en el paso 3)")

print("Entorno listo." if fotos else "Faltan las fotos de recursos/.")
