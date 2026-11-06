"""Prueba del entorno · Práctica 1

Comprueba que Python, NumPy, OpenCV y Matplotlib están instalados y que se pueden abrir ventanas.
Ejecutar (con el entorno virtual activado):  python prueba_entorno.py
"""
import sys                        # información del intérprete de Python

import cv2                        # OpenCV (paquete opencv-python)
import numpy as np                # NumPy: arreglos numéricos
import matplotlib                 # Matplotlib: gráficas y figuras
import matplotlib.pyplot as plt   # interfaz de Matplotlib para dibujar figuras

# 1) Versiones instaladas: si alguna importación falla, el script se detiene antes de llegar aquí
print("Python    :", sys.version.split()[0])
print("NumPy     :", np.__version__)
print("OpenCV    :", cv2.__version__)
print("Matplotlib:", matplotlib.__version__)

# 2) Una imagen hecha a mano: 100 filas x 300 columnas x 3 canales, todos los píxeles en 0 (negro)
imagen = np.zeros((100, 300, 3), dtype=np.uint8)
imagen[:, 0:100] = (255, 0, 0)    # primer tercio: azul  (OpenCV guarda los colores en orden B, G, R)
imagen[:, 100:200] = (0, 255, 0)  # segundo tercio: verde
imagen[:, 200:300] = (0, 0, 255)  # último tercio: rojo
print("Forma de la imagen:", imagen.shape)

# 3) Mostrarla con OpenCV: la ventana espera a que se pulse una tecla
cv2.imshow("Prueba OpenCV (pulsa una tecla)", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4) Mostrarla con Matplotlib: espera colores en orden R, G, B, así que se convierte primero
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title("Prueba Matplotlib (cierra la ventana)")
plt.show()

print("Entorno listo.")
