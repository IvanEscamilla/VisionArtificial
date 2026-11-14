"""Práctica 3 · Paso 5 · Movimiento
===================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Detectar lo que se mueve con tres modelos: diferencia de frames, fondo
promedio y MOG2, y ver cómo lo que se detiene termina siendo fondo.

Ejecutar
--------
    python p3_paso5_movimiento.py
    python p3_paso5_movimiento.py --video
    python p3_paso5_movimiento.py --video --guardar
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from p3_paso1_umbral import limpiar, porcentaje, preparar_gris


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
# Funciones nuevas del paso 5
# =====================================================================================
def diferencia_frames(gris_anterior, gris, umbral=25):
    """Objeto = píxeles que cambiaron más de 'umbral' entre dos frames seguidos."""
    _, mascara = cv2.threshold(cv2.absdiff(gris, gris_anterior), umbral, 255, cv2.THRESH_BINARY)
    return mascara


class FondoPromedio:
    """Fondo = promedio móvil de los frames: fondo = (1 - alfa)·fondo + alfa·frame."""

    def __init__(self):
        self.fondo = None                                     # float32: acumular en uint8 perdería decimales

    def aplicar(self, gris, alfa=0.02, umbral=25):
        if self.fondo is None:
            self.fondo = gris.astype(np.float32)              # el primer frame es el fondo inicial
        cv2.accumulateWeighted(gris, self.fondo, alfa)        # alfa chica = aprende lento
        fondo_8 = cv2.convertScaleAbs(self.fondo)             # a uint8 para restarlo y verlo
        _, mascara = cv2.threshold(cv2.absdiff(gris, fondo_8), umbral, 255, cv2.THRESH_BINARY)
        return mascara, fondo_8


def cajas_en_movimiento(mascara, area_min=300):
    """Rectángulos de las manchas de movimiento con área suficiente."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contornos if cv2.contourArea(c) >= area_min]


def dibujar_cajas(frame, cajas):
    """Dibuja rectángulos verdes sobre una copia del frame."""
    salida = frame.copy()
    for x, y, w, h in cajas:
        cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 255, 0), 2)
    escribir(salida, f"objetos en movimiento: {len(cajas)}")
    return salida


# =====================================================================================
# Modos de ejecución
# =====================================================================================
def modo_interactivo():
    """Barras de alfa, umbral y área mínima; mosaico 2×3 de los tres modelos."""
    captura, es_archivo, espera = abrir_fuente()
    VENTANA = "Paso 5 - Movimiento"
    cv2.namedWindow(VENTANA)
    barra("Alfa x1000", VENTANA, 20, 100)
    barra("Umbral", VENTANA, 25, 100)
    barra("Area min", VENTANA, 300, 5000)

    promedio = FondoPromedio()
    mog2 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    gris_anterior = None
    ultimo = None

    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        alfa = valor("Alfa x1000", VENTANA) / 1000
        umbral = valor("Umbral", VENTANA)
        area_min = valor("Area min", VENTANA)

        gris = preparar_gris(frame, 1.5)
        diff = (diferencia_frames(gris_anterior, gris, umbral)
                if gris_anterior is not None else np.zeros_like(gris))
        prom, fondo_prom = promedio.aplicar(gris, alfa, umbral)
        mog = mog2.apply(gris)
        fondo_mog = mog2.getBackgroundImage()
        if fondo_mog is None:
            fondo_mog = np.zeros_like(gris)
        elif fondo_mog.ndim == 3:
            fondo_mog = cv2.cvtColor(fondo_mog, cv2.COLOR_BGR2GRAY)

        diff, prom, mog = limpiar(diff, 5), limpiar(prom, 5), limpiar(mog, 5)
        cajas = cajas_en_movimiento(mog, area_min)
        vista = dibujar_cajas(frame, cajas)
        ultimo = (diff, prom, mog, cajas)

        titulos = [
            f"Diferencia de frames\n{porcentaje(diff):.2f} %",
            f"Fondo promedio α={alfa:.3f}\n{porcentaje(prom):.2f} %",
            f"MOG2\n{porcentaje(mog):.2f} %",
            "Fondo promedio",
            "Fondo de MOG2",
            f"Movimiento MOG2 · {len(cajas)} cajas",
        ]
        cv2.imshow(VENTANA, mosaico([diff, prom, mog, fondo_prom, fondo_mog, vista],
                                    titulos, columnas=3, ancho=400))
        gris_anterior = gris.copy()

        if (cv2.waitKey(espera) & 0xFF) in (27, ord("q")):
            break

    captura.release()
    cv2.destroyAllWindows()
    if ultimo:
        diff, prom, mog, cajas = ultimo
        print("== Último frame ==")
        print(f"diferencia   {porcentaje(diff):5.2f} % de la imagen en movimiento")
        print(f"promedio     {porcentaje(prom):5.2f} % de la imagen en movimiento")
        print(f"MOG2         {porcentaje(mog):5.2f} % de la imagen en movimiento")
        print(f"objetos en movimiento (MOG2): {len(cajas)}")


def procesar_series(alfa=0.02, umbral=25, area_min=300):
    """Recorre los 300 frames y devuelve las series de porcentaje de movimiento."""
    captura = cv2.VideoCapture(str(RUTA_VIDEO))
    promedio = FondoPromedio()
    mog2 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    gris_anterior = None
    series = {"frame": [], "diferencia": [], "promedio": [], "mog2": [], "cajas": []}
    frames_guardados = {}
    n = 0
    while True:
        ok, frame = captura.read()
        if not ok:
            break
        gris = preparar_gris(frame, 1.5)
        diff = (diferencia_frames(gris_anterior, gris, umbral)
                if gris_anterior is not None else np.zeros_like(gris))
        prom, fondo_prom = promedio.aplicar(gris, alfa, umbral)
        mog = mog2.apply(gris)
        fondo_mog = mog2.getBackgroundImage()
        if fondo_mog is None:
            fondo_mog = np.zeros_like(gris)
        elif fondo_mog.ndim == 3:
            fondo_mog = cv2.cvtColor(fondo_mog, cv2.COLOR_BGR2GRAY)
        diff, prom, mog = limpiar(diff, 5), limpiar(prom, 5), limpiar(mog, 5)
        cajas = cajas_en_movimiento(mog, area_min)

        series["frame"].append(n)
        series["diferencia"].append(porcentaje(diff))
        series["promedio"].append(porcentaje(prom))
        series["mog2"].append(porcentaje(mog))
        series["cajas"].append(len(cajas))

        if n in (100, 230) or n == 0:
            frames_guardados[n] = (frame.copy(), diff.copy(), prom.copy(), mog.copy(),
                                   fondo_prom.copy(), fondo_mog.copy(), cajas[:])
        gris_anterior = gris.copy()
        n += 1
    captura.release()
    return series, frames_guardados


def modo_guardar():
    """Genera las capturas en los frames 100 y 230 y la gráfica temporal de las capturas."""
    CARPETA_CAPTURAS.mkdir(exist_ok=True)
    series, frames = procesar_series(0.02, 25, 300)

    for n in (100, 230):
        frame, diff, prom, mog, fondo_prom, fondo_mog, cajas = frames[n]
        print(f"== Frame {n} ==")
        print(f"diferencia   {porcentaje(diff):5.2f} % de la imagen en movimiento")
        print(f"promedio     {porcentaje(prom):5.2f} % de la imagen en movimiento")
        print(f"MOG2         {porcentaje(mog):5.2f} % de la imagen en movimiento")
        print(f"objetos en movimiento (MOG2): {len(cajas)}")
        titulos = [
            f"Diferencia\n{porcentaje(diff):.2f} %",
            f"Fondo promedio α=0.02\n{porcentaje(prom):.2f} %",
            f"MOG2\n{porcentaje(mog):.2f} %",
            "Fondo promedio",
            "Fondo de MOG2",
            f"Movimiento MOG2 · {len(cajas)} cajas",
        ]
        cv2.imwrite(str(CARPETA_CAPTURAS / f"p3_paso5_frame{n}.png"),
                    mosaico([diff, prom, mog, fondo_prom, fondo_mog, dibujar_cajas(frame, cajas)],
                            titulos, columnas=3, ancho=420))

    # Series temporales: cuándo aparece y desaparece cada objeto
    with open(CARPETA_CAPTURAS / "p3_paso5_series.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "diferencia_pct", "promedio_pct", "mog2_pct", "cajas_mog2"])
        for i in range(len(series["frame"])):
            w.writerow([series["frame"][i], f"{series['diferencia'][i]:.3f}",
                        f"{series['promedio'][i]:.3f}", f"{series['mog2'][i]:.3f}", series["cajas"][i]])

    fig, ejes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    fig.suptitle("Paso 5 · Porcentaje de la imagen en movimiento a lo largo del video",
                 fontsize=13, fontweight="bold")
    ejes[0].plot(series["frame"], series["diferencia"], label="Diferencia de frames", color="#d62728")
    ejes[0].plot(series["frame"], series["promedio"], label="Fondo promedio α=0.02", color="#1f77b4")
    ejes[0].plot(series["frame"], series["mog2"], label="MOG2", color="#2ca02c")
    ejes[0].set_ylabel("% de píxeles en movimiento")
    ejes[0].legend()
    ejes[0].grid(alpha=0.3)
    ejes[1].plot(series["frame"], series["cajas"], color="#2ca02c")
    ejes[1].set_ylabel("cajas MOG2 (área ≥ 300)")
    ejes[1].set_xlabel("frame")
    ejes[1].grid(alpha=0.3)
    fig.savefig(CARPETA_CAPTURAS / "p3_paso5_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Efecto de alfa: valores típicos en el frame 230
    print("\n== Efecto de alfa en el frame 230 ==")
    print(f"{'alfa':>8}  {'promedio %':>11}")
    for alfa in (0.001, 0.02, 0.1):
        s, _ = procesar_series(alfa, 25, 300)
        print(f"{alfa:>8.3f}  {s['promedio'][230]:>11.2f}")

    print("\n[capturas] capturas y series guardadas en reporte/p3_paso5_*.png y p3_paso5_series.csv")


if __name__ == "__main__":
    if MODO_GUARDAR:
        modo_guardar()
    else:
        modo_interactivo()
