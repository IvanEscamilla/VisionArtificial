"""Práctica 3 · Paso 2 · Segmentación por fronteras
===================================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Medir los cambios bruscos de intensidad con Sobel y Canny, y convertir
bordes en regiones cerrando huecos y rellenando contornos.

Ejecutar
--------
    python p3_paso2_bordes.py
    python p3_paso2_bordes.py --video
    python p3_paso2_bordes.py --video --guardar
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from p3_paso1_umbral import preparar_gris, porcentaje       # reutilizamos el paso 1 (mismo folder)


# =====================================================================================
# [BASE] Video: abrir la fuente, leer frames, medir FPS, escribir texto, mosaico y barras
# =====================================================================================
CARPETA = Path(__file__).resolve().parent
RUTA_VIDEO = CARPETA / "recursos" / "video_sintetico.mp4"
SALIDAS = CARPETA / "salidas"
CARPETA_CAPTURAS = CARPETA / "reporte"
MODO_GUARDAR = "--guardar" in sys.argv


def abrir_fuente():
    """Webcam por defecto; con --video usa RUTA_VIDEO (o la ruta que siga a --video)."""
    if "--video" in sys.argv:
        i = sys.argv.index("--video")
        siguiente = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        ruta = siguiente if siguiente and not siguiente.startswith("--") else RUTA_VIDEO
        captura = cv2.VideoCapture(str(ruta))
        es_archivo = True
    else:
        captura = cv2.VideoCapture(0)
        es_archivo = False
    if not captura.isOpened():
        sys.exit("No se pudo abrir la fuente de video (¿cámara ocupada? prueba con --video)")
    fps = captura.get(cv2.CAP_PROP_FPS) or 30
    espera = max(1, int(1000 / fps)) if es_archivo else 1
    return captura, es_archivo, espera


def leer_frame(captura, es_archivo):
    """Lee el siguiente frame. Un archivo vuelve a empezar al terminar."""
    ok, frame = captura.read()
    if not ok and es_archivo:
        captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = captura.read()
    return frame if ok else None


class MedidorFPS:
    """Mide los cuadros por segundo reales (promedio suavizado)."""

    def __init__(self):
        self.anterior = time.perf_counter()
        self.fps = 0.0

    def actualizar(self):
        ahora = time.perf_counter()
        dt = ahora - self.anterior
        self.anterior = ahora
        if dt > 0:
            self.fps = 1 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt
        return self.fps


def escribir(img, texto, fila=0, color=(255, 255, 255)):
    """Escribe texto sobre un rectángulo negro para que se lea siempre."""
    y = 22 + 24 * fila
    (w, h), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (4, y - h - 5), (12 + w, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, texto, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def mosaico(imagenes, titulos, columnas=3, ancho=400):
    """Une varias imágenes en una rejilla con título en cada celda."""
    celdas = []
    for img, titulo in zip(imagenes, titulos):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alto = int(img.shape[0] * ancho / img.shape[1])
        celda = cv2.resize(img, (ancho, alto))
        escribir(celda, titulo, color=(0, 255, 255))
        celdas.append(celda)
    while len(celdas) % columnas:
        celdas.append(np.zeros_like(celdas[-1]))
    filas = [np.hstack(celdas[i:i + columnas]) for i in range(0, len(celdas), columnas)]
    return np.vstack(filas)


def barra(nombre, ventana, inicial, maximo):
    """Crea una barra deslizante en una ventana que ya existe."""
    cv2.createTrackbar(nombre, ventana, inicial, maximo, lambda v: None)


def valor(nombre, ventana):
    """Lee el valor actual de una barra deslizante."""
    return cv2.getTrackbarPos(nombre, ventana)


# =====================================================================================
# Funciones nuevas del paso 2
# =====================================================================================
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
    cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, elemento)   # une bordes casi tocándose
    contornos, _ = cv2.findContours(cerrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # solo los de afuera
    regiones = np.zeros_like(gris)
    cv2.drawContours(regiones, contornos, -1, 255, cv2.FILLED)        # -1 = todos; FILLED = rellenos
    return bordes, cerrados, regiones


# =====================================================================================
# Modos de ejecución
# =====================================================================================
def modo_interactivo():
    """Barras de suavizado, Canny bajo/alto y cierre; mosaico de 6 paneles."""
    captura, es_archivo, espera = abrir_fuente()
    VENTANA = "Paso 2 - Bordes"
    cv2.namedWindow(VENTANA)
    barra("Suavizado x10", VENTANA, 15, 50)
    barra("Canny bajo", VENTANA, 50, 255)
    barra("Canny alto", VENTANA, 150, 255)
    barra("Cierre", VENTANA, 7, 21)

    ultimo = None
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        sigma = valor("Suavizado x10", VENTANA) / 10
        bajo = valor("Canny bajo", VENTANA)
        alto = valor("Canny alto", VENTANA)
        k = valor("Cierre", VENTANA)
        if k % 2 == 0:
            k += 1

        gris = preparar_gris(frame, sigma)
        gx, gy, magnitud = gradiente_sobel(gris)
        bordes, cerrados, regiones = segmentar_bordes(gris, bajo, alto, k)
        ultimo = (bordes, regiones)

        titulos = [
            "|Gx| · bordes verticales",
            "|Gy| · bordes horizontales",
            "Magnitud √(Gx²+Gy²)",
            f"Canny {bajo}/{alto} · {int((bordes > 0).sum())} px",
            f"Bordes cerrados k={k}",
            f"Regiones rellenas · {porcentaje(regiones):.1f} %",
        ]
        cv2.imshow(VENTANA, mosaico([a_8bits(gx), a_8bits(gy), a_8bits(magnitud),
                                      bordes, cerrados, regiones], titulos, columnas=3, ancho=400))
        if (cv2.waitKey(espera) & 0xFF) in (27, ord("q")):
            break

    captura.release()
    cv2.destroyAllWindows()
    if ultimo:
        bordes, regiones = ultimo
        print("== Último frame ==")
        print(f"píxeles de borde (Canny): {int((bordes > 0).sum())}")
        print(f"regiones rellenas       : {porcentaje(regiones):.1f} % de la imagen")


def modo_guardar():
    """Genera el mosaico y las comparaciones de Canny de las capturas (frame 160)."""
    captura, es_archivo, espera = abrir_fuente()
    frame = None
    for _ in range(161):
        frame = leer_frame(captura, es_archivo)
    captura.release()
    if frame is None:
        sys.exit("No se pudo leer el frame de trabajo")

    CARPETA_CAPTURAS.mkdir(exist_ok=True)
    gris = preparar_gris(frame, 1.5)
    gx, gy, magnitud = gradiente_sobel(gris)
    bordes, cerrados, regiones = segmentar_bordes(gris, 50, 150, 7)

    print("== Último frame ==")
    print(f"píxeles de borde (Canny): {int((bordes > 0).sum())}")
    print(f"regiones rellenas       : {porcentaje(regiones):.1f} % de la imagen")

    titulos = [
        "|Gx| · bordes verticales",
        "|Gy| · bordes horizontales",
        "Magnitud √(Gx²+Gy²)",
        f"Canny 50/150 · {int((bordes > 0).sum())} px",
        "Bordes cerrados k=7",
        f"Regiones rellenas · {porcentaje(regiones):.1f} %",
    ]
    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso2_mosaico.png"),
                mosaico([a_8bits(gx), a_8bits(gy), a_8bits(magnitud), bordes, cerrados, regiones],
                        titulos, columnas=3, ancho=420))

    # Comparaciones para las preguntas: sin suavizado y con umbrales distintos
    gris_sin = preparar_gris(frame, 0)
    b_sin, _, _ = segmentar_bordes(gris_sin, 50, 150, 7)
    b_bajo, _, _ = segmentar_bordes(gris, 20, 80, 7)
    b_alto, _, _ = segmentar_bordes(gris, 100, 200, 7)
    casos = [("Sin suavizado (σ=0)", b_sin), ("Canny 50/150 (σ=1.5)", bordes),
             ("Canny 20/80", b_bajo), ("Canny 100/200", b_alto)]
    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso2_comparacion.png"),
                mosaico([m for _, m in casos],
                        [f"{n} · {int((m > 0).sum())} px" for n, m in casos],
                        columnas=2, ancho=520))
    print("\n[capturas] mosaicos guardados en reporte/p3_paso2_*.png")


if __name__ == "__main__":
    if MODO_GUARDAR:
        modo_guardar()
    else:
        modo_interactivo()
