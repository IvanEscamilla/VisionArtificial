"""Práctica 3 · Paso 7 · Comparativa con IoU
=============================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Aplicar todas las técnicas de segmentación de los pasos 1–6 a los 300
frames del video sintético y compararlas con la máscara correcta usando
IoU y Dice, en dos tareas:
    A. Todos los objetos contra el fondo.
    B. Solo la pelota naranja.

Ejecutar
--------
    python p3_paso7_comparativa.py
    python p3_paso7_comparativa.py --guardar
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

from p3_paso1_umbral import (limpiar, preparar_gris, segmentar_adaptativo,
                             segmentar_global, segmentar_otsu)
from p3_paso2_bordes import segmentar_bordes
from p3_paso3_agrupacion import crecer_region, grupo_de_la_semilla, segmentar_kmeans
from p3_paso4_dominio import mascara_color, segmentar_dominio
from p3_paso5_movimiento import FondoPromedio
from p3_paso6_hough_snake import circulos_hough


# =====================================================================================
# Configuración
# =====================================================================================
CARPETA = Path(__file__).resolve().parent
RUTA_VIDEO = CARPETA / "recursos" / "video_sintetico.mp4"
RUTA_MASCARAS = CARPETA / "recursos" / "mascaras.npz"
CARPETA_CAPTURAS = CARPETA / "reporte"
CALENTAMIENTO = 30                                        # frames para que los modelos de fondo aprendan
MODO_GUARDAR = "--guardar" in sys.argv


def iou(a, b):
    """Intersección sobre unión de dos máscaras 0/255."""
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 1.0


def dice(a, b):
    """Coeficiente Dice de dos máscaras 0/255."""
    a, b = a > 0, b > 0
    total = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / total if total else 1.0


def centro_de(mascara):
    """Centro (x, y) de una máscara, con momentos de imagen. Hace las veces del clic del usuario."""
    m = cv2.moments(mascara, binaryImage=True)
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def circulo_mas_cercano(gris, semilla):
    """Máscara del círculo de Hough más cercano a la semilla (vacía si no hay círculos)."""
    mascara = np.zeros_like(gris)
    circulos = circulos_hough(gris)
    if circulos:
        x, y, r = min(circulos, key=lambda c: (c[0] - semilla[0]) ** 2 + (c[1] - semilla[1]) ** 2)
        cv2.circle(mascara, (x, y), r, 255, -1)
    return mascara


def medir(funcion, *argumentos):
    """Ejecuta funcion(*argumentos) y devuelve (resultado, milisegundos)."""
    t0 = time.perf_counter()
    resultado = funcion(*argumentos)
    return resultado, (time.perf_counter() - t0) * 1000


def mosaico(imagenes, titulos, columnas=3, ancho=400):
    """Une varias imágenes en una rejilla; cada celda lleva un encabezado negro."""
    celdas = []
    for img, titulo in zip(imagenes, titulos):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alto = int(img.shape[0] * ancho / img.shape[1])
        img = cv2.resize(img, (ancho, alto))
        encabezado = np.zeros((26, ancho, 3), np.uint8)
        cv2.putText(encabezado, titulo, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        celdas.append(np.vstack([encabezado, img]))
    while len(celdas) % columnas:
        celdas.append(np.zeros_like(celdas[-1]))
    filas = [np.hstack(celdas[i:i + columnas]) for i in range(0, len(celdas), columnas)]
    return np.vstack(filas)


# =====================================================================================
# Análisis principal
# =====================================================================================
def analizar():
    """Recorre el video, aplica todas las técnicas y calcula IoU/Dice/tiempo."""
    verdad = dict(np.load(RUTA_MASCARAS))                 # descomprimir una sola vez (si no, tarda minutos)
    captura = cv2.VideoCapture(str(RUTA_VIDEO))
    if not captura.isOpened():
        sys.exit(f"No se pudo abrir el video: {RUTA_VIDEO}")

    fondo = FondoPromedio()
    mog2 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

    # Cada método guarda su lista de IoU, Dice y tiempos
    metodos_a = ["Global T=150", "Otsu", "Adaptativo + limpieza", "Bordes (Canny) rellenos",
                 "Fondo promedio", "MOG2", "Saturación HSV (dominio)"]
    metodos_b = ["Grupo k-means", "Crecimiento de región", "Color naranja",
                 "Color + tamaño + forma", "Hough círculo"]
    resultados = {m: {"iou": [], "dice": [], "ms": []} for m in metodos_a + metodos_b}
    mascaras_200 = None
    n = 0

    while True:
        ok, frame = captura.read()
        if not ok:
            break
        gris = preparar_gris(frame, 1.5)
        objetos = verdad["objetos"][n]
        pelota = verdad["pelota"][n]
        semilla = centro_de(pelota)

        # ---------------- Tarea A: todos los objetos contra el fondo ----------------
        global_, ms = medir(segmentar_global, gris, 150)
        resultados["Global T=150"]["ms"].append(ms)

        (otsu, _), ms = medir(segmentar_otsu, gris)
        resultados["Otsu"]["ms"].append(ms)

        adapt, ms = medir(lambda: limpiar(segmentar_adaptativo(gris, 101, 10), 5))
        resultados["Adaptativo + limpieza"]["ms"].append(ms)

        (_, _, regiones_bordes), ms = medir(segmentar_bordes, gris, 50, 150, 7)
        resultados["Bordes (Canny) rellenos"]["ms"].append(ms)

        (prom, _), ms = medir(fondo.aplicar, gris, 0.02, 25)
        prom = limpiar(prom, 5)
        resultados["Fondo promedio"]["ms"].append(ms)

        def aplicar_mog():
            return mog2.apply(gris)
        mog, ms = medir(aplicar_mog)
        mog = limpiar(mog, 5)
        resultados["MOG2"]["ms"].append(ms)

        def hsv_saturacion():
            return limpiar(mascara_color(frame, 0, 179, 80, 0), 5)
        hsv_dom, ms = medir(hsv_saturacion)
        resultados["Saturación HSV (dominio)"]["ms"].append(ms)

        # ---------------- Tarea B: solo la pelota naranja ----------------
        (etiquetas, _), ms = medir(segmentar_kmeans, frame, 4, 0.25)
        kmeans_mask = grupo_de_la_semilla(etiquetas, semilla)
        resultados["Grupo k-means"]["ms"].append(ms)

        region, ms = medir(crecer_region, frame, semilla, 30)
        resultados["Crecimiento de región"]["ms"].append(ms)

        naranja, ms = medir(mascara_color, frame, 5, 25, 120, 80)
        resultados["Color naranja"]["ms"].append(ms)

        (_, dominio, _, _), ms = medir(segmentar_dominio, frame, (5, 25, 120, 80), 1500, 0.90, 5)
        resultados["Color + tamaño + forma"]["ms"].append(ms)

        hough_mask, ms = medir(circulo_mas_cercano, gris, semilla)
        resultados["Hough círculo"]["ms"].append(ms)

        # ---------------- Comparación con la verdad ----------------
        if n >= CALENTAMIENTO:
            pares_a = [("Global T=150", global_), ("Otsu", otsu),
                       ("Adaptativo + limpieza", adapt), ("Bordes (Canny) rellenos", regiones_bordes),
                       ("Fondo promedio", prom), ("MOG2", mog),
                       ("Saturación HSV (dominio)", hsv_dom)]
            pares_b = [("Grupo k-means", kmeans_mask), ("Crecimiento de región", region),
                       ("Color naranja", naranja), ("Color + tamaño + forma", dominio),
                       ("Hough círculo", hough_mask)]
            for nombre, mascara in pares_a:
                resultados[nombre]["iou"].append(iou(mascara, objetos))
                resultados[nombre]["dice"].append(dice(mascara, objetos))
            for nombre, mascara in pares_b:
                resultados[nombre]["iou"].append(iou(mascara, pelota))
                resultados[nombre]["dice"].append(dice(mascara, pelota))

        if n == 200:
            mascaras_200 = {
                "frame": frame.copy(), "objetos": objetos.copy(), "pelota": pelota.copy(),
                "global": global_.copy(), "otsu": otsu.copy(), "adapt": adapt.copy(),
                "bordes": regiones_bordes.copy(), "prom": prom.copy(), "mog": mog.copy(),
                "hsv": hsv_dom.copy(), "kmeans": kmeans_mask.copy(), "region": region.copy(),
                "naranja": naranja.copy(), "dominio": dominio.copy(), "hough": hough_mask.copy(),
            }
        n += 1

    captura.release()
    if n == 0:
        sys.exit("El video no entregó ningún frame")

    # ---------------- Tablas e impresión ----------------
    print(f"== Resultados en {n - CALENTAMIENTO} frames (del {CALENTAMIENTO} al {n - 1}) ==")
    print("\nA. Todos los objetos")
    print(f"{'método':<30} {'IoU':>6} {'Dice':>6} {'ms/frame':>9}")
    for m in metodos_a:
        print(f"{m:<30} {np.mean(resultados[m]['iou']):>6.3f} {np.mean(resultados[m]['dice']):>6.3f} "
              f"{np.mean(resultados[m]['ms']):>9.2f}")
    print("\nB. Solo la pelota naranja")
    print(f"{'método':<30} {'IoU':>6} {'Dice':>6} {'ms/frame':>9}")
    for m in metodos_b:
        print(f"{m:<30} {np.mean(resultados[m]['iou']):>6.3f} {np.mean(resultados[m]['dice']):>6.3f} "
              f"{np.mean(resultados[m]['ms']):>9.2f}")

    # ---------------- Guardar CSV ----------------
    CARPETA_CAPTURAS.mkdir(exist_ok=True)
    with open(CARPETA_CAPTURAS / "p3_iou.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tarea", "metodo", "IoU", "Dice", "ms_frame"])
        for tarea, metodos in (("A", metodos_a), ("B", metodos_b)):
            for m in metodos:
                w.writerow([tarea, m, f"{np.mean(resultados[m]['iou']):.4f}",
                            f"{np.mean(resultados[m]['dice']):.4f}",
                            f"{np.mean(resultados[m]['ms']):.4f}"])

    # ---------------- Figura de barras ----------------
    fig, ejes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    fig.suptitle("Paso 7 · IoU promedio de cada técnica de segmentación", fontsize=14, fontweight="bold")
    for eje, titulo, metodos in ((ejes[0], "A. Todos los objetos", metodos_a),
                                 (ejes[1], "B. Solo la pelota naranja", metodos_b)):
        valores = [np.mean(resultados[m]["iou"]) for m in metodos]
        eje.barh(metodos, valores, color="#2ca02c")
        eje.set_xlim(0, 1)
        eje.set_xlabel("IoU promedio")
        eje.set_title(titulo)
        eje.invert_yaxis()
        eje.grid(axis="x", alpha=0.3)
        for i, v in enumerate(valores):
            eje.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    fig.savefig(CARPETA_CAPTURAS / "p3_paso7_iou.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Máscaras del frame 200 ----------------
    if mascaras_200 is not None:
        m = mascaras_200
        paneles = [
            m["frame"], m["objetos"], m["global"], m["otsu"], m["adapt"], m["bordes"],
            m["prom"], m["mog"], m["hsv"], m["pelota"], m["kmeans"], m["region"],
            m["naranja"], m["dominio"], m["hough"],
        ]
        titulos = [
            "Original (frame 200)", "Verdad: objetos", "Global T=150", "Otsu",
            "Adaptativo + limpieza", "Bordes rellenos", "Fondo promedio", "MOG2",
            "Saturación HSV", "Verdad: pelota", "Grupo k-means", "Crecimiento de región",
            "Color naranja", "Color + tamaño + forma", "Hough círculo",
        ]
        cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso7_mascaras200.png"),
                    mosaico(paneles, titulos, columnas=3, ancho=400))

    print("\n[capturas] tabla y figuras guardadas en reporte/p3_iou.csv, p3_paso7_iou.png y p3_paso7_mascaras200.png")

    if not MODO_GUARDAR:
        plt.show()


if __name__ == "__main__":
    analizar()
