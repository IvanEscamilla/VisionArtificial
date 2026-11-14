"""Práctica 3 · Paso 2 · Segmentación por fronteras
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado):
    python p3_paso2_bordes.py                  # webcam
    python p3_paso2_bordes.py --video          # recursos/video_sintetico.mp4

Barras: Suavizado · Canny bajo · Canny alto · Cierre
Teclas: Esc o Q = salir
"""
import sys                        # sys.argv (argumentos) y sys.exit (terminar con un mensaje)
import time                       # time.perf_counter(): reloj de alta precisión para medir FPS y ms
from pathlib import Path          # rutas que funcionan igual en Windows, macOS y Linux

import cv2                        # OpenCV: cámara, video, filtros, segmentación y ventanas
import numpy as np                # NumPy: cada frame es un arreglo de NumPy

# =====================================================================================
# [BASE] Video: abrir la fuente, leer frames, medir FPS, escribir texto, mosaico y barras
# =====================================================================================
CARPETA = Path(__file__).resolve().parent                     # carpeta donde está este script
RUTA_VIDEO = CARPETA / "recursos" / "video_sintetico.mp4"     # video de respaldo (sin cámara)
SALIDAS = CARPETA / "salidas"                                 # aquí se guardan capturas y grabaciones


def abrir_fuente():
    """Webcam por defecto; con --video usa RUTA_VIDEO (o la ruta que siga a --video).
    Devuelve (captura, es_archivo, espera_ms)."""
    if "--video" in sys.argv:                                 # ¿se pidió un archivo de video?
        i = sys.argv.index("--video")                         # posición de "--video" en la lista
        ruta = sys.argv[i + 1] if i + 1 < len(sys.argv) else RUTA_VIDEO   # ruta escrita o la de respaldo
        captura = cv2.VideoCapture(str(ruta))                 # abrir el archivo
        es_archivo = True
    else:
        captura = cv2.VideoCapture(0)                         # 0 = primera cámara del equipo
        es_archivo = False
    if not captura.isOpened():                                # no existe, está ocupada o el códec falla
        sys.exit("No se pudo abrir la fuente de video (¿cámara ocupada? prueba con --video)")
    fps = captura.get(cv2.CAP_PROP_FPS) or 30                 # algunos drivers reportan 0 -> usamos 30
    espera = max(1, int(1000 / fps)) if es_archivo else 1     # un archivo se reproduce a su velocidad real
    return captura, es_archivo, espera


def leer_frame(captura, es_archivo):
    """Lee el siguiente frame. Un archivo vuelve a empezar al terminar. Devuelve None si no hay imagen."""
    ok, frame = captura.read()                                # ok = True si llegó un frame
    if not ok and es_archivo:                                 # fin del archivo: regresamos al frame 0
        captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = captura.read()
    return frame if ok else None


class MedidorFPS:
    """Mide los cuadros por segundo reales (promedio suavizado)."""

    def __init__(self):
        self.anterior = time.perf_counter()                    # instante del frame anterior
        self.fps = 0.0

    def actualizar(self):
        ahora = time.perf_counter()
        dt = ahora - self.anterior                            # segundos entre este frame y el anterior
        self.anterior = ahora
        if dt > 0:
            self.fps = 1 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt   # suaviza los saltos
        return self.fps


def escribir(img, texto, fila=0, color=(255, 255, 255)):
    """Escribe texto en la esquina superior izquierda, sobre un rectángulo negro para que se lea siempre."""
    y = 22 + 24 * fila                                        # cada fila de texto baja 24 px
    (w, h), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)   # tamaño que ocupará el texto
    cv2.rectangle(img, (4, y - h - 5), (12 + w, y + base + 2), (0, 0, 0), -1) # -1 = rectángulo relleno
    cv2.putText(img, texto, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def mosaico(imagenes, titulos, columnas=3, ancho=400):
    """Une varias imágenes (color o gris) en una sola, en una rejilla con título en cada celda."""
    celdas = []
    for img, titulo in zip(imagenes, titulos):
        if img.ndim == 2:                                     # gris o máscara -> 3 canales para poder unirla
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alto = int(img.shape[0] * ancho / img.shape[1])       # conservar la proporción
        celda = cv2.resize(img, (ancho, alto))                # todas las celdas del mismo tamaño
        escribir(celda, titulo, color=(0, 255, 255))          # título en amarillo
        celdas.append(celda)
    while len(celdas) % columnas:                             # completar la última fila con celdas negras
        celdas.append(np.zeros_like(celdas[-1]))
    filas = [np.hstack(celdas[i:i + columnas]) for i in range(0, len(celdas), columnas)]   # pegar en horizontal
    return np.vstack(filas)                                   # y luego las filas en vertical


def barra(nombre, ventana, inicial, maximo):
    """Crea una barra deslizante (trackbar) en una ventana que ya existe."""
    cv2.createTrackbar(nombre, ventana, inicial, maximo, lambda v: None)   # lambda: no hacer nada al moverla


def valor(nombre, ventana):
    """Lee el valor actual de una barra deslizante."""
    return cv2.getTrackbarPos(nombre, ventana)


# =====================================================================================
# Funciones de segmentación por fronteras
# =====================================================================================
from p3_paso1_umbral import preparar_gris                     # reutilizamos el paso 1 (mismo folder)


def gradiente_sobel(gris):
    """Derivadas con Sobel. CV_32F: las derivadas pueden ser negativas y uint8 no las guarda."""
    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)           # cambio de izquierda a derecha (bordes verticales)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)           # cambio de arriba a abajo (bordes horizontales)
    magnitud = cv2.magnitude(gx, gy)                          # sqrt(gx² + gy²): fuerza del borde
    return gx, gy, magnitud


def a_8bits(m):
    """Lleva cualquier matriz (float, con negativos) a 0..255 para poder verla."""
    return cv2.normalize(np.abs(m), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def segmentar_bordes(gris, bajo=50, alto=150, k_cierre=7):
    """Canny -> cerrar huecos -> rellenar contornos externos. Devuelve (bordes, cerrados, regiones)."""
    bordes = cv2.Canny(gris, bajo, alto)                      # 255 en los bordes, 0 en el resto
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_cierre, k_cierre))
    cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, elemento)        # une bordes casi tocándose
    contornos, _ = cv2.findContours(cerrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # solo los de afuera
    regiones = np.zeros_like(gris)
    cv2.drawContours(regiones, contornos, -1, 255, cv2.FILLED)            # -1 = todos; FILLED = rellenos
    return bordes, cerrados, regiones


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    captura, es_archivo, espera = abrir_fuente()

    VENTANA = "Paso 2 - Bordes"
    cv2.namedWindow(VENTANA)
    barra("Suavizado x10", VENTANA, 15, 50)
    barra("Canny bajo", VENTANA, 50, 255)
    barra("Canny alto", VENTANA, 150, 255)
    barra("Cierre", VENTANA, 7, 15)

    print("Mueve las barras. Esc o Q = salir")
    ultimo = None

    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break

        gris = preparar_gris(frame, valor("Suavizado x10", VENTANA) / 10)
        bajo = valor("Canny bajo", VENTANA)
        alto = valor("Canny alto", VENTANA)
        k = max(3, valor("Cierre", VENTANA))

        gx, gy, magnitud = gradiente_sobel(gris)
        bordes, cerrados, regiones = segmentar_bordes(gris, bajo, alto, k)

        pixeles_borde = cv2.countNonZero(bordes)
        porcentaje = 100 * cv2.countNonZero(regiones) / regiones.size
        panel = mosaico(
            [a_8bits(gx), a_8bits(gy), a_8bits(magnitud), bordes, cerrados, regiones],
            ["|Gx| · bordes verticales", "|Gy| · bordes horizontales", "Magnitud √(Gx²+Gy²)",
             f"Canny ({bajo}, {alto}) · {pixeles_borde} px",
             f"Bordes cerrados (k={k})", f"Regiones rellenas · {porcentaje:.1f} %"],
            columnas=3, ancho=400)
        cv2.imshow(VENTANA, panel)

        ultimo = (pixeles_borde, porcentaje)
        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break

    if ultimo is not None:
        print("\n== Último frame ==")
        print(f"píxeles de borde (Canny): {ultimo[0]}")
        print(f"regiones rellenas       : {ultimo[1]:.1f} % de la imagen")

    captura.release()
    cv2.destroyAllWindows()
