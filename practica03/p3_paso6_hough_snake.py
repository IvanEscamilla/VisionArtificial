"""Práctica 3 · Paso 6 · Hough y contornos activos
==================================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Detectar líneas y círculos con la transformada de Hough y ajustar un
contorno activo (snake) al borde de un objeto.

Ejecutar
--------
    python p3_paso6_hough_snake.py
    python p3_paso6_hough_snake.py --video
    python p3_paso6_hough_snake.py --video --guardar
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import gaussian                    # scikit-image: suavizado para la snake
from skimage.segmentation import active_contour          # scikit-image: contorno activo (snake)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from p3_paso1_umbral import preparar_gris


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
# Funciones nuevas del paso 6
# =====================================================================================
def lineas_hough(gris, votos=80, largo_min=100):
    """Segmentos de recta: [(x1, y1, x2, y2), ...]."""
    bordes = cv2.Canny(gris, 50, 150)                         # Hough trabaja sobre los bordes
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, votos,   # resolución: 1 px y 1 grado
                             minLineLength=largo_min, maxLineGap=10)
    return [] if lineas is None else [tuple(int(v) for v in l) for l in lineas.reshape(-1, 4)]


def circulos_hough(gris, sensibilidad=30, r_min=15, r_max=60):
    """Círculos: [(x, y, r), ...]. Sensibilidad = votos mínimos del acumulador."""
    suave = cv2.medianBlur(gris, 5)
    circulos = cv2.HoughCircles(suave, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                                param1=120, param2=sensibilidad, minRadius=r_min, maxRadius=r_max)
    return [] if circulos is None else [tuple(int(v) for v in c)
                                        for c in np.round(circulos.reshape(-1, 3))]


def snake(gris, centro, radio, alfa=0.015, beta=10.0, gamma=0.001, puntos=200):
    """Contorno activo que empieza como un círculo. scikit-image usa coordenadas (fila, columna).

    alfa = elasticidad (que no se estire) · beta = rigidez (que no se doble) ·
    gamma = tamaño del paso.
    """
    t = np.linspace(0, 2 * np.pi, puntos)
    inicial = np.column_stack([centro[1] + radio * np.sin(t), centro[0] + radio * np.cos(t)])   # (y, x)
    suave = gaussian(gris, sigma=3, preserve_range=False)     # la snake necesita bordes suaves para "sentirlos"
    final = active_contour(suave, inicial, alpha=alfa, beta=beta, gamma=gamma)
    return inicial, final


def a_puntos_cv(curva):
    """(fila, columna) de scikit-image -> (x, y) enteros de OpenCV para dibujar con polylines."""
    return np.round(curva[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)


def radio_medio(curva, centro):
    """Radio medio de una curva (fila, columna) respecto a un centro (x, y)."""
    return float(np.mean(np.sqrt((curva[:, 0] - centro[1]) ** 2 + (curva[:, 1] - centro[0]) ** 2)))


# =====================================================================================
# Modos de ejecución
# =====================================================================================
def modo_interactivo():
    """Ventana con Hough en vivo; un clic congela el frame y lanza la snake."""
    captura, es_archivo, espera = abrir_fuente()
    VENTANA = "Paso 6 - Hough"
    VENTANA_SNAKE = "Paso 6 - Snake"
    cv2.namedWindow(VENTANA)
    cv2.namedWindow(VENTANA_SNAKE)
    barra("Votos", VENTANA, 80, 200)
    barra("Largo min", VENTANA, 100, 300)
    barra("Sensibilidad", VENTANA, 30, 100)
    barra("Radio snake", VENTANA, 60, 150)
    barra("Alfa x1000", VENTANA, 15, 100)
    barra("Beta", VENTANA, 10, 50)

    estado = {"frame": None, "centro": None}

    def al_hacer_clic(evento, x, y, banderas, parametro):
        if evento == cv2.EVENT_LBUTTONDOWN and estado["frame"] is not None:
            estado["centro"] = (x, y)
            frame = estado["frame"].copy()
            gris = preparar_gris(frame, 1.5)
            radio = valor("Radio snake", VENTANA)
            alfa = valor("Alfa x1000", VENTANA) / 1000
            beta = valor("Beta", VENTANA)
            t0 = time.perf_counter()
            inicial, final = snake(gris, (x, y), radio, alfa, beta)
            ms = (time.perf_counter() - t0) * 1000
            cv2.polylines(frame, [a_puntos_cv(inicial)], True, (0, 255, 255), 1)
            cv2.polylines(frame, [a_puntos_cv(final)], True, (0, 255, 0), 2)
            cv2.imshow(VENTANA_SNAKE, frame)
            cv2.waitKey(1)
            print(f"snake en ({x}, {y}): {ms:.0f} ms, radio medio final = {radio_medio(final, (x, y)):.1f} px")

    cv2.setMouseCallback(VENTANA, al_hacer_clic)
    print("Haz clic en el centro de un objeto para lanzar la snake.")

    ultimo = None
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        estado["frame"] = frame.copy()
        gris = preparar_gris(frame, 1.5)
        lineas = lineas_hough(gris, max(1, valor("Votos", VENTANA)), valor("Largo min", VENTANA))
        circulos = circulos_hough(gris, max(1, valor("Sensibilidad", VENTANA)), 15, 150)

        resultado = frame.copy()
        for x1, y1, x2, y2 in lineas:
            cv2.line(resultado, (x1, y1), (x2, y2), (0, 0, 255), 1)
        for x, y, r in circulos:
            cv2.circle(resultado, (x, y), r, (0, 255, 0), 2)
        escribir(resultado, f"líneas: {len(lineas)}   círculos: {len(circulos)}")
        cv2.imshow(VENTANA, resultado)
        ultimo = (lineas, circulos)

        if (cv2.waitKey(espera) & 0xFF) in (27, ord("q")):
            break

    captura.release()
    cv2.destroyAllWindows()
    if ultimo:
        lineas, circulos = ultimo
        print("== Último frame ==")
        print("líneas  :", lineas)
        print("círculos:", circulos)


def modo_guardar():
    """Genera las capturas de Hough y de la snake de las capturas (frame 160)."""
    captura, es_archivo, espera = abrir_fuente()
    frame = None
    for _ in range(161):
        frame = leer_frame(captura, es_archivo)
    captura.release()
    if frame is None:
        sys.exit("No se pudo leer el frame de trabajo")

    CARPETA_CAPTURAS.mkdir(exist_ok=True)
    gris = preparar_gris(frame, 1.5)
    lineas = lineas_hough(gris, 80, 100)
    circulos = circulos_hough(gris, 30, 15, 60)

    print("== Último frame ==")
    print("líneas  :", lineas)
    print("círculos:", circulos)

    resultado = frame.copy()
    for x1, y1, x2, y2 in lineas:
        cv2.line(resultado, (x1, y1), (x2, y2), (0, 0, 255), 1)
    for x, y, r in circulos:
        cv2.circle(resultado, (x, y), r, (0, 255, 0), 2)
    escribir(resultado, f"líneas: {len(lineas)}   círculos: {len(circulos)}")
    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso6_hough.png"), resultado)

    # Snake sobre la pelota naranja
    centro = (576, 268)
    t0 = time.perf_counter()
    inicial, final = snake(gris, centro, 60, 0.015, 10.0)
    ms = (time.perf_counter() - t0) * 1000
    radio = radio_medio(final, centro)
    print(f"snake en {centro}: {ms:.0f} ms, radio medio final = {radio:.1f} px")

    con_snake = frame.copy()
    cv2.polylines(con_snake, [a_puntos_cv(inicial)], True, (0, 255, 255), 1)
    cv2.polylines(con_snake, [a_puntos_cv(final)], True, (0, 255, 0), 2)
    cv2.drawMarker(con_snake, centro, (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
    escribir(con_snake, f"snake: {ms:.0f} ms · radio medio {radio:.1f} px")
    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso6_snake.png"), con_snake)

    # Variantes para las preguntas: alfa, beta, radio y centro
    variantes = [
        ("Base α=0.015 β=10 r=60", centro, 60, 0.015, 10.0),
        ("α=0.1", centro, 60, 0.1, 10.0),
        ("β=0", centro, 60, 0.015, 0.0),
        ("r=30 (no encierra)", centro, 30, 0.015, 10.0),
        ("centro desplazado +80 px", (centro[0] + 80, centro[1]), 60, 0.015, 10.0),
        ("r=90", centro, 90, 0.015, 10.0),
    ]
    paneles, titulos = [], []
    for titulo, c, r, a, b in variantes:
        ini, fin = snake(gris, c, r, a, b)
        panel = frame.copy()
        cv2.polylines(panel, [a_puntos_cv(ini)], True, (0, 255, 255), 1)
        cv2.polylines(panel, [a_puntos_cv(fin)], True, (0, 255, 0), 2)
        cv2.drawMarker(panel, c, (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
        paneles.append(panel)
        titulos.append(f"{titulo}\nradio medio {radio_medio(fin, c):.1f} px")
        print(f"{titulo:<28} radio medio final = {radio_medio(fin, c):.1f} px")
    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso6_variantes.png"),
                mosaico(paneles, titulos, columnas=3, ancho=420))
    print("\n[capturas] capturas guardadas en reporte/p3_paso6_*.png")


if __name__ == "__main__":
    if MODO_GUARDAR:
        modo_guardar()
    else:
        modo_interactivo()
