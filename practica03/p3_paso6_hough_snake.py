"""Práctica 3 · Paso 6 · Hough y contornos activos
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado):
    python p3_paso6_hough_snake.py                  # webcam
    python p3_paso6_hough_snake.py --video          # recursos/video_sintetico.mp4

Haz clic en el centro de un objeto para lanzar la snake.
Barras: Votos · Largo min · Sensibilidad · Radio snake · Alfa x1000 · Beta · Teclas: Esc o Q = salir
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
# Funciones de Hough y contornos activos
# =====================================================================================
from p3_paso1_umbral import preparar_gris
from skimage.filters import gaussian                          # scikit-image: suavizado para la snake
from skimage.segmentation import active_contour               # scikit-image: contorno activo (snake)


def lineas_hough(gris, votos=80, largo_min=100):
    """Segmentos de recta: [(x1, y1, x2, y2), ...]."""
    bordes = cv2.Canny(gris, 50, 150)                         # Hough trabaja sobre los bordes
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, votos,   # resolución: 1 px y 1 grado
                             minLineLength=largo_min, maxLineGap=10)
    return [] if lineas is None else [tuple(int(v) for v in l) for l in lineas.reshape(-1, 4)]   # una fila por línea


def circulos_hough(gris, sensibilidad=30, r_min=15, r_max=60):
    """Círculos: [(x, y, r), ...]. Sensibilidad = votos mínimos del acumulador (más bajo = más círculos)."""
    suave = cv2.medianBlur(gris, 5)
    circulos = cv2.HoughCircles(suave, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                                param1=120, param2=sensibilidad, minRadius=r_min, maxRadius=r_max)
    return [] if circulos is None else [tuple(int(v) for v in c) for c in np.round(circulos.reshape(-1, 3))]


def snake(gris, centro, radio, alfa=0.015, beta=10.0, gamma=0.001, puntos=200):
    """Contorno activo que empieza como un círculo. scikit-image usa coordenadas (fila, columna).
    alfa = elasticidad (que no se estire) · beta = rigidez (que no se doble) · gamma = tamaño del paso."""
    t = np.linspace(0, 2 * np.pi, puntos)
    inicial = np.column_stack([centro[1] + radio * np.sin(t), centro[0] + radio * np.cos(t)])   # (y, x)
    suave = gaussian(gris, sigma=3, preserve_range=False)     # la snake necesita bordes suaves para "sentirlos"
    final = active_contour(suave, inicial, alpha=alfa, beta=beta, gamma=gamma)
    return inicial, final


def a_puntos_cv(curva):
    """(fila, columna) de scikit-image -> (x, y) enteros de OpenCV para dibujar con polylines."""
    return np.round(curva[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    captura, es_archivo, espera = abrir_fuente()

    VENTANA = "Paso 6 - Hough"
    cv2.namedWindow(VENTANA)
    barra("Votos", VENTANA, 80, 200)
    barra("Largo min", VENTANA, 100, 400)
    barra("Sensibilidad", VENTANA, 30, 100)
    barra("Radio snake", VENTANA, 60, 150)
    barra("Alfa x1000", VENTANA, 15, 100)
    barra("Beta", VENTANA, 10, 50)

    estado = {"frame": None}

    def al_hacer_clic(evento, x, y, banderas, parametro):
        if evento == cv2.EVENT_LBUTTONDOWN and estado["frame"] is not None:
            radio = valor("Radio snake", VENTANA)
            alfa = valor("Alfa x1000", VENTANA) / 1000
            beta = valor("Beta", VENTANA)
            gris = cv2.cvtColor(estado["frame"], cv2.COLOR_BGR2GRAY)

            t0 = time.perf_counter()
            inicial, final = snake(gris, (x, y), radio, alfa, beta)
            ms = (time.perf_counter() - t0) * 1000

            # Radio medio final: distancia de cada punto al centro del clic
            distancias = np.hypot(final[:, 1] - x, final[:, 0] - y)
            print(f"snake en ({x}, {y}): {ms:.0f} ms, radio medio final = {distancias.mean():.1f} px")

            resultado = estado["frame"].copy()
            cv2.polylines(resultado, [a_puntos_cv(inicial)], True, (0, 255, 255), 1)
            cv2.polylines(resultado, [a_puntos_cv(final)], True, (0, 255, 0), 2)
            cv2.circle(resultado, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("Paso 6 - Snake (cualquier tecla)", resultado)
            cv2.waitKey(0)

    cv2.setMouseCallback(VENTANA, al_hacer_clic)
    print("Haz clic en el centro de un objeto para lanzar la snake. Esc o Q = salir")

    ultimo = None
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        estado["frame"] = frame

        gris = preparar_gris(frame, 1.5)
        votos = valor("Votos", VENTANA)
        largo_min = valor("Largo min", VENTANA)
        sensibilidad = valor("Sensibilidad", VENTANA)

        lineas = lineas_hough(gris, votos, largo_min)
        circulos = circulos_hough(gris, sensibilidad, 15, 60)

        resultado = frame.copy()
        for x1, y1, x2, y2 in lineas:
            cv2.line(resultado, (x1, y1), (x2, y2), (0, 0, 255), 1)
        for x, y, r in circulos:
            cv2.circle(resultado, (x, y), r, (0, 255, 0), 2)
            cv2.circle(resultado, (x, y), 2, (0, 255, 0), -1)

        escribir(resultado, f"líneas: {len(lineas)} · círculos: {len(circulos)}")
        cv2.imshow(VENTANA, resultado)

        ultimo = (lineas, circulos)
        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break

    if ultimo is not None:
        lineas, circulos = ultimo
        print("\n== Último frame ==")
        print("líneas :", lineas)
        print("círculos:", circulos)

    captura.release()
    cv2.destroyAllWindows()
