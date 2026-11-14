"""Práctica 3 · Paso 7 · Comparativa con IoU
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado), desde la carpeta practica03:
    python p3_paso7_comparativa.py

Recorre los 300 frames del video sintético, aplica cada técnica y la compara con
recursos/mascaras.npz (la segmentación correcta) usando IoU y Dice.
"""
import sys                        # sys.argv (argumentos) y sys.exit (terminar con un mensaje)
import time                       # time.perf_counter(): reloj de alta precisión para medir FPS y ms
from pathlib import Path          # rutas que funcionan igual en Windows, macOS y Linux

import cv2                        # OpenCV: cámara, video, filtros, segmentación y ventanas
import numpy as np                # NumPy: cada frame es un arreglo de NumPy
import matplotlib.pyplot as plt   # gráficas de resultados

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
# Técnicas importadas de los pasos anteriores
# =====================================================================================
from p3_paso1_umbral import (limpiar, preparar_gris, segmentar_adaptativo, segmentar_global,
                             segmentar_otsu)
from p3_paso2_bordes import segmentar_bordes
from p3_paso3_agrupacion import crecer_region, grupo_de_la_semilla, segmentar_kmeans
from p3_paso4_dominio import mascara_color, segmentar_dominio
from p3_paso5_movimiento import FondoPromedio
from p3_paso6_hough_snake import circulos_hough

RUTA_MASCARAS = CARPETA / "recursos" / "mascaras.npz"
CALENTAMIENTO = 30                                            # frames para que los modelos de fondo aprendan


def iou(a, b):
    """Intersección sobre unión de dos máscaras 0/255."""
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 1.0


def dice(a, b):
    """Coeficiente de Dice: 2·|A∩B| / (|A| + |B|)."""
    a, b = a > 0, b > 0
    total = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / total if total else 1.0


def centro_de(mascara):
    """Centro (x, y) de una máscara, con momentos de imagen. Hace las veces del clic del usuario."""
    m = cv2.moments(mascara, binaryImage=True)
    if m["m00"] == 0:
        return mascara.shape[1] // 2, mascara.shape[0] // 2
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def circulo_mas_cercano(gris, semilla):
    """Máscara del círculo de Hough más cercano a la semilla (vacía si no hay círculos)."""
    mascara = np.zeros_like(gris)
    circulos = circulos_hough(gris)
    if circulos:
        x, y, r = min(circulos, key=lambda c: (c[0] - semilla[0]) ** 2 + (c[1] - semilla[1]) ** 2)
        cv2.circle(mascara, (x, y), r, 255, -1)
    return mascara


def mascara_saturacion(frame, s_min=100):
    """Todos los objetos del video son saturados; el fondo gris tiene S baja. Máscara 0/255."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return limpiar(cv2.inRange(hsv, (0, s_min, 0), (179, 255, 255)), 5)


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    verdad = dict(np.load(RUTA_MASCARAS))                     # descomprimir UNA sola vez
    captura = cv2.VideoCapture(str(RUTA_VIDEO))               # la comparación siempre usa el video de referencia
    if not captura.isOpened():
        sys.exit(f"No se pudo leer el video: {RUTA_VIDEO}")

    metodos_a = ["Global T=150", "Otsu", "Adaptativo + limpieza", "Bordes (Canny) rellenos",
                 "Fondo promedio", "MOG2", "Saturación HSV (dominio)"]
    metodos_b = ["Grupo k-means", "Crecimiento de región", "Color naranja",
                 "Color + tamaño + forma", "Hough círculo"]

    valores_a = {m: {"iou": [], "dice": [], "ms": []} for m in metodos_a}
    valores_b = {m: {"iou": [], "dice": [], "ms": []} for m in metodos_b}

    promedio = FondoPromedio()
    mog2 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    mascaras_200 = {}
    frame_200 = None

    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    for n in range(total):
        frame = leer_frame(captura, True)
        if frame is None:
            break

        gris = preparar_gris(frame, 1.5)
        evaluar = n >= CALENTAMIENTO

        # ---------- Tarea A: todos los objetos ----------
        resultado_a = {}

        t0 = time.perf_counter(); resultado_a["Global T=150"] = segmentar_global(gris, 150)
        ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter(); resultado_a["Otsu"] = segmentar_otsu(gris)[0]
        ms_o = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        resultado_a["Adaptativo + limpieza"] = limpiar(segmentar_adaptativo(gris, 101, 10), 5)
        ms_ad = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter(); resultado_a["Bordes (Canny) rellenos"] = segmentar_bordes(gris, 50, 150, 7)[2]
        ms_b = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        mascara_prom, _ = promedio.aplicar(gris, 0.02, 25)
        resultado_a["Fondo promedio"] = limpiar(mascara_prom, 5)
        ms_p = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        resultado_a["MOG2"] = limpiar(mog2.apply(gris), 5)
        ms_m = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter(); resultado_a["Saturación HSV (dominio)"] = mascara_saturacion(frame, 100)
        ms_s = (time.perf_counter() - t0) * 1000

        tiempos_a = {"Global T=150": ms, "Otsu": ms_o, "Adaptativo + limpieza": ms_ad,
                     "Bordes (Canny) rellenos": ms_b, "Fondo promedio": ms_p, "MOG2": ms_m,
                     "Saturación HSV (dominio)": ms_s}

        # ---------- Tarea B: solo la pelota naranja ----------
        semilla = centro_de(verdad["pelota"][n])
        resultado_b = {}

        t0 = time.perf_counter()
        etiquetas, _ = segmentar_kmeans(frame, 4, 0.25)
        resultado_b["Grupo k-means"] = grupo_de_la_semilla(etiquetas, semilla)
        ms_k = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter(); resultado_b["Crecimiento de región"] = crecer_region(frame, semilla, 30)
        ms_c = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter(); resultado_b["Color naranja"] = mascara_color(frame, 5, 25, 120, 80)
        ms_co = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        resultado_b["Color + tamaño + forma"] = segmentar_dominio(
            frame, (5, 25, 120, 80), area_min=1500, circ_min=0.90, k=5)[1]
        ms_d = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter(); resultado_b["Hough círculo"] = circulo_mas_cercano(gris, semilla)
        ms_h = (time.perf_counter() - t0) * 1000

        tiempos_b = {"Grupo k-means": ms_k, "Crecimiento de región": ms_c,
                     "Color naranja": ms_co, "Color + tamaño + forma": ms_d,
                     "Hough círculo": ms_h}

        if evaluar:
            for metodo in metodos_a:
                valores_a[metodo]["iou"].append(iou(resultado_a[metodo], verdad["objetos"][n]))
                valores_a[metodo]["dice"].append(dice(resultado_a[metodo], verdad["objetos"][n]))
                valores_a[metodo]["ms"].append(tiempos_a[metodo])
            for metodo in metodos_b:
                valores_b[metodo]["iou"].append(iou(resultado_b[metodo], verdad["pelota"][n]))
                valores_b[metodo]["dice"].append(dice(resultado_b[metodo], verdad["pelota"][n]))
                valores_b[metodo]["ms"].append(tiempos_b[metodo])

        if n == 200:
            frame_200 = frame.copy()
            mascaras_200 = {**resultado_a, **resultado_b,
                            "verdad objetos": verdad["objetos"][n],
                            "verdad pelota": verdad["pelota"][n]}

    captura.release()

    # ---------- Tablas ----------
    print(f"\n== Resultados en {len(next(iter(valores_a.values()))['iou'])} frames "
          f"(del {CALENTAMIENTO} al {total - 1}) ==")
    print("\nA. Todos los objetos")
    print(f"{'método':<30} {'IoU':>6} {'Dice':>6} {'ms/frame':>9}")
    resumen_a = {}
    for metodo in metodos_a:
        v = valores_a[metodo]
        resumen_a[metodo] = (np.mean(v["iou"]), np.mean(v["dice"]), np.mean(v["ms"]))
        print(f"{metodo:<30} {np.mean(v['iou']):>6.3f} {np.mean(v['dice']):>6.3f} "
              f"{np.mean(v['ms']):>9.2f}")

    print("\nB. Solo la pelota naranja")
    print(f"{'método':<30} {'IoU':>6} {'Dice':>6} {'ms/frame':>9}")
    resumen_b = {}
    for metodo in metodos_b:
        v = valores_b[metodo]
        resumen_b[metodo] = (np.mean(v["iou"]), np.mean(v["dice"]), np.mean(v["ms"]))
        print(f"{metodo:<30} {np.mean(v['iou']):>6.3f} {np.mean(v['dice']):>6.3f} "
              f"{np.mean(v['ms']):>9.2f}")

    # ---------- Figuras ----------
    fig1, ejes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained")
    fig1.suptitle("Práctica 3 · IoU promedio de cada técnica", fontsize=14)
    for eje, metodos, resumen, titulo in [
            (ejes[0], metodos_a, resumen_a, "A. Todos los objetos"),
            (ejes[1], metodos_b, resumen_b, "B. Solo la pelota naranja")]:
        valores = [resumen[m][0] for m in metodos]
        barras = eje.barh(metodos, valores, color="tab:blue")
        eje.set_xlim(0, 1)
        eje.set_xlabel("IoU promedio (1 = perfecto)")
        eje.set_title(titulo)
        eje.invert_yaxis()
        eje.bar_label(barras, fmt="%.3f", padding=2, fontsize=8)

    def figura_mascaras(titulo, metodos, mascaras, verdad_clave):
        columnas = 3
        filas = int(np.ceil((len(metodos) + 2) / columnas))
        fig, ejes = plt.subplots(filas, columnas, figsize=(13, 4.2 * filas), layout="constrained")
        fig.suptitle(titulo, fontsize=14)
        paneles = [("Original", frame_200), ("Verdad", mascaras[verdad_clave])]
        paneles += [(m, mascaras[m]) for m in metodos]
        for eje in ejes.flat:
            eje.axis("off")
        for eje, (nombre, img) in zip(ejes.flat, paneles):
            if nombre == "Original":
                eje.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                eje.imshow(img, cmap="gray", vmin=0, vmax=255)
            ref = mascaras[verdad_clave]
            texto = nombre if nombre in ("Original", "Verdad") else f"{nombre}\nIoU={iou(img, ref):.3f}"
            eje.set_title(texto, fontsize=9)

    figura_mascaras("Máscaras en el frame 200 · A. Todos los objetos", metodos_a, mascaras_200,
                    "verdad objetos")
    figura_mascaras("Máscaras en el frame 200 · B. Solo la pelota", metodos_b, mascaras_200,
                    "verdad pelota")

    plt.show()
