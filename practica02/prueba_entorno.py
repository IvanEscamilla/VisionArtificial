"""Prueba del entorno · Práctica 2

Comprueba que las librerías están instaladas (incluida scikit-image, que usará la Práctica 3),
que se puede leer el video de recursos/ y que la webcam abre.
Ejecutar (con el entorno virtual activado):  python prueba_entorno.py
"""
import sys                        # información del intérprete de Python
from pathlib import Path          # rutas que funcionan en Windows, macOS y Linux

import cv2                        # OpenCV (paquete opencv-python)
import numpy as np                # NumPy: arreglos numéricos
import matplotlib                 # Matplotlib: gráficas y figuras
import skimage                    # scikit-image (paquete scikit-image)

RUTA_VIDEO = Path(__file__).resolve().parent / "recursos" / "video_sintetico.mp4"

# 1) Versiones instaladas: si alguna importación falla, el script se detiene antes de llegar aquí
print("Python      :", sys.version.split()[0])
print("NumPy       :", np.__version__)
print("OpenCV      :", cv2.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("scikit-image:", skimage.__version__)

# 2) El video de respaldo: se abre, se leen sus propiedades y su primer frame
video = cv2.VideoCapture(str(RUTA_VIDEO))
ok_video, frame_video = video.read()
if ok_video:
    ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video       : {ancho}x{alto}, {video.get(cv2.CAP_PROP_FPS):.0f} FPS, "
          f"{int(video.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
else:
    print("Video       : NO se pudo leer", RUTA_VIDEO, "(revisa la carpeta recursos/)")
video.release()

# 3) La webcam: se abre, se toma un frame y se libera
camara = cv2.VideoCapture(0)
ok_camara, frame_camara = camara.read() if camara.isOpened() else (False, None)
camara.release()
if ok_camara:
    print(f"Webcam      : {frame_camara.shape[1]}x{frame_camara.shape[0]}")
else:
    print("Webcam      : no disponible (cierra Zoom/Teams o usa --video en los scripts)")

# 4) Mostrar lo que se pudo leer; las ventanas esperan a que se pulse una tecla
if ok_video:
    cv2.imshow("Video de recursos (pulsa una tecla)", frame_video)
if ok_camara:
    cv2.imshow("Webcam (pulsa una tecla)", frame_camara)
if ok_video or ok_camara:
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Entorno listo." if ok_video else "Falta el video de recursos/.")
