"""Práctica 3 · Paso 1 · Umbralización
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado):
    python p3_paso1_umbral.py                  # webcam
    python p3_paso1_umbral.py --video          # recursos/video_sintetico.mp4

Barras: Suavizado · T global · Bloque · C · Limpieza
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
# Funciones de segmentación por umbral
# =====================================================================================
def preparar_gris(frame, sigma=1.5):
    """Gris suavizado: el ruido hace que píxeles vecinos caigan de lados distintos del umbral."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gris, (0, 0), sigma) if sigma > 0 else gris


def segmentar_global(gris, t):
    """Objeto = píxeles MÁS OSCUROS que t (INV invierte: oscuro -> 255). Un solo T para toda la imagen."""
    _, mascara = cv2.threshold(gris, t, 255, cv2.THRESH_BINARY_INV)
    return mascara


def segmentar_otsu(gris):
    """Otsu elige T solo: el que mejor separa el histograma en dos grupos. Devuelve (máscara, T)."""
    t, mascara = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # el 0 se ignora
    return mascara, t


def segmentar_adaptativo(gris, bloque=101, c=10):
    """Un T distinto por píxel: el promedio (ponderado) de su vecindad de bloque x bloque menos c."""
    return cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bloque, c)


def limpiar(mascara, k=5):
    """Apertura (erosión + dilatación) quita lo más delgado que k; cierre (dilatación + erosión) tapa huecos."""
    if k < 2:
        return mascara
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))     # "pincel" circular de k x k
    abierta = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemento)
    return cv2.morphologyEx(abierta, cv2.MORPH_CLOSE, elemento)


def porcentaje(mascara):
    """Qué porcentaje de la imagen quedó como objeto (255)."""
    return 100 * cv2.countNonZero(mascara) / mascara.size


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    captura, es_archivo, espera = abrir_fuente()

    VENTANA = "Paso 1 - Umbral"
    cv2.namedWindow(VENTANA)
    barra("Suavizado x10", VENTANA, 15, 50)      # sigma = valor / 10
    barra("T global", VENTANA, 150, 255)
    barra("Bloque", VENTANA, 50, 100)            # bloque = 2*valor + 1
    barra("C", VENTANA, 10, 50)
    barra("Limpieza", VENTANA, 5, 15)

    print("Mueve las barras. Esc o Q = salir")
    ultimo = None

    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break

        gris = preparar_gris(frame, valor("Suavizado x10", VENTANA) / 10)
        t_global = valor("T global", VENTANA)
        bloque = 2 * max(1, valor("Bloque", VENTANA)) + 1        # impar y > 1
        c = valor("C", VENTANA)
        k = valor("Limpieza", VENTANA)

        global_ = segmentar_global(gris, t_global)
        otsu, t_otsu = segmentar_otsu(gris)
        adapt = segmentar_adaptativo(gris, bloque, c)
        limpia = limpiar(adapt, k)

        panel = mosaico(
            [frame, gris, global_, otsu, adapt, limpia],
            [f"Original", f"Gris suavizado (σ={valor('Suavizado x10', VENTANA) / 10:.1f})",
             f"Global T={t_global} · {porcentaje(global_):.1f} %",
             f"Otsu T={t_otsu:.0f} · {porcentaje(otsu):.1f} %",
             f"Adaptativo · {porcentaje(adapt):.1f} %",
             f"Adaptativo limpio · {porcentaje(limpia):.1f} %"],
            columnas=3, ancho=400)
        cv2.imshow(VENTANA, panel)

        ultimo = (t_global, t_otsu, porcentaje(global_), porcentaje(otsu),
                  porcentaje(adapt), porcentaje(limpia))
        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break

    if ultimo is not None:
        t_global, t_otsu, p_global, p_otsu, p_adapt, p_limpia = ultimo
        print("\n== Último frame ==")
        print(f"T global = {t_global}  T de Otsu = {t_otsu:.0f}")
        print(f"global      objeto = {p_global:.1f} % de la imagen")
        print(f"otsu        objeto = {p_otsu:.1f} % de la imagen")
        print(f"adaptativo  objeto = {p_adapt:.1f} % de la imagen")
        print(f"limpia      objeto = {p_limpia:.1f} % de la imagen")

    captura.release()
    cv2.destroyAllWindows()
