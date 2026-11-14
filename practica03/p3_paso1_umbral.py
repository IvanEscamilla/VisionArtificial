"""Práctica 3 · Paso 1 · Umbralización
=====================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Comparar el umbral global, el de Otsu y el adaptativo, y limpiar la máscara
con morfología (apertura + cierre).

Ejecutar
--------
    python p3_paso1_umbral.py
    python p3_paso1_umbral.py --video
    python p3_paso1_umbral.py --video --guardar
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


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
# Funciones de segmentación del paso 1
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
    t, mascara = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mascara, t


def segmentar_adaptativo(gris, bloque=101, c=10):
    """Un T distinto por píxel: el promedio (ponderado) de su vecindad de bloque x bloque menos c."""
    return cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, bloque, c)


def limpiar(mascara, k=5):
    """Apertura (erosión + dilatación) quita lo más delgado que k; cierre tapa huecos."""
    if k < 2:
        return mascara
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))     # "pincel" circular de k x k
    abierta = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemento)
    return cv2.morphologyEx(abierta, cv2.MORPH_CLOSE, elemento)


def porcentaje(mascara):
    """Qué porcentaje de la imagen quedó como objeto (255)."""
    return 100 * cv2.countNonZero(mascara) / mascara.size


# =====================================================================================
# Modos de ejecución
# =====================================================================================
def modo_interactivo():
    """Ventana con 5 barras y mosaico 2×3: original, gris, global, Otsu, adaptativo y limpio."""
    captura, es_archivo, espera = abrir_fuente()
    VENTANA = "Paso 1 - Umbralizacion"
    cv2.namedWindow(VENTANA)
    barra("Suavizado x10", VENTANA, 15, 50)       # sigma = valor/10
    barra("T global", VENTANA, 150, 255)
    barra("Bloque", VENTANA, 50, 75)              # bloque = 2·valor + 1
    barra("C", VENTANA, 10, 30)
    barra("Limpieza", VENTANA, 5, 15)

    ultimo = None
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        sigma = valor("Suavizado x10", VENTANA) / 10
        t = valor("T global", VENTANA)
        bloque = 2 * max(1, valor("Bloque", VENTANA)) + 1     # impar y > 1
        c = valor("C", VENTANA)
        k = valor("Limpieza", VENTANA)

        gris = preparar_gris(frame, sigma)
        global_ = segmentar_global(gris, t)
        otsu, t_otsu = segmentar_otsu(gris)
        adapt = segmentar_adaptativo(gris, bloque, c)
        limpia = limpiar(adapt, k)
        ultimo = (t, t_otsu, global_, otsu, adapt, limpia)

        titulos = [
            "Original",
            f"Gris suavizado σ={sigma:.1f}",
            f"Global T={t}\nobjeto {porcentaje(global_):4.1f} %",
            f"Otsu T={int(t_otsu)}\nobjeto {porcentaje(otsu):4.1f} %",
            f"Adaptativo bloque={bloque} C={c}\nobjeto {porcentaje(adapt):4.1f} %",
            f"Adaptativo limpio k={k}\nobjeto {porcentaje(limpia):4.1f} %",
        ]
        cv2.imshow(VENTANA, mosaico([frame, gris, global_, otsu, adapt, limpia],
                                    titulos, columnas=3, ancho=400))
        if (cv2.waitKey(espera) & 0xFF) in (27, ord("q")):
            break

    captura.release()
    cv2.destroyAllWindows()
    if ultimo:
        t, t_otsu, global_, otsu, adapt, limpia = ultimo
        print("== Último frame ==")
        print(f"T global = {t}   T de Otsu = {int(t_otsu)}")
        for nombre, m in (("global", global_), ("otsu", otsu), ("adaptativo", adapt), ("limpia", limpia)):
            print(f"{nombre:<12} objeto = {porcentaje(m):5.1f} % de la imagen")


def modo_guardar():
    """Genera el mosaico y las comparaciones de umbrales de las capturas (frame 160)."""
    captura, es_archivo, espera = abrir_fuente()
    frame = None
    for _ in range(161):
        frame = leer_frame(captura, es_archivo)
    captura.release()
    if frame is None:
        sys.exit("No se pudo leer el frame de trabajo")

    CARPETA_CAPTURAS.mkdir(exist_ok=True)
    gris = preparar_gris(frame, 1.5)
    global_ = segmentar_global(gris, 150)
    otsu, t_otsu = segmentar_otsu(gris)
    adapt = segmentar_adaptativo(gris, 101, 10)
    limpia = limpiar(adapt, 5)

    print("== Último frame ==")
    print(f"T global = 150   T de Otsu = {int(t_otsu)}")
    for nombre, m in (("global", global_), ("otsu", otsu), ("adaptativo", adapt), ("limpia", limpia)):
        print(f"{nombre:<12} objeto = {porcentaje(m):5.1f} % de la imagen")

    titulos = [
        "Original",
        "Gris suavizado σ=1.5",
        f"Global T=150 · {porcentaje(global_):.1f} %",
        f"Otsu T={int(t_otsu)} · {porcentaje(otsu):.1f} %",
        f"Adaptativo 101×101 C=10 · {porcentaje(adapt):.1f} %",
        f"Adaptativo limpio k=5 · {porcentaje(limpia):.1f} %",
    ]
    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso1_mosaico.png"),
                mosaico([frame, gris, global_, otsu, adapt, limpia], titulos, columnas=3, ancho=420))

    # Comparaciones para responder las preguntas: T global 100/150/200, bloques 5 y 151
    casos = [
        ("Global T=100", segmentar_global(gris, 100)),
        ("Global T=150", global_),
        ("Global T=200", segmentar_global(gris, 200)),
        ("Otsu", otsu),
        ("Adaptativo bloque=5", segmentar_adaptativo(gris, 5, 10)),
        ("Adaptativo bloque=151", segmentar_adaptativo(gris, 151, 10)),
        ("Adaptativo limpio k=9", limpiar(adapt, 9)),
    ]
    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso1_comparacion.png"),
                mosaico([m for _, m in casos],
                        [f"{n} · {porcentaje(m):.1f} %" for n, m in casos],
                        columnas=3, ancho=420))
    print("\n[capturas] mosaicos guardados en reporte/p3_paso1_*.png")


if __name__ == "__main__":
    if MODO_GUARDAR:
        modo_guardar()
    else:
        modo_interactivo()
