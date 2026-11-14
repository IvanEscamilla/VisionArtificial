"""Práctica 3 · Paso 3 · Agrupación de píxeles
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado):
    python p3_paso3_agrupacion.py                  # webcam
    python p3_paso3_agrupacion.py --video          # recursos/video_sintetico.mp4

Haz clic sobre un objeto en el panel "Original" para tomar la semilla.
Barras: K · Tolerancia · Teclas: Esc o Q = salir
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
# Funciones de agrupación
# =====================================================================================
def segmentar_kmeans(frame, k=4, escala=0.25):
    """k-means sobre los colores (B, G, R) de una versión reducida del frame (para que corra en vivo).
    Devuelve (etiquetas, pintada): el grupo de cada píxel y la imagen pintada con el color de su grupo."""
    chica = cv2.resize(frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    datos = chica.reshape(-1, 3).astype(np.float32)           # una fila por píxel: [B, G, R]
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)   # 10 iteraciones o cambio < 1
    _, etiquetas, centros = cv2.kmeans(datos, k, None, criterio, 1, cv2.KMEANS_PP_CENTERS)
    etiquetas = etiquetas.reshape(chica.shape[:2])            # de lista a matriz: grupo de cada píxel
    pintada = centros.astype(np.uint8)[etiquetas]             # cada píxel toma el color (centro) de su grupo
    alto, ancho = frame.shape[:2]
    return (cv2.resize(etiquetas.astype(np.uint8), (ancho, alto), interpolation=cv2.INTER_NEAREST),
            cv2.resize(pintada, (ancho, alto), interpolation=cv2.INTER_NEAREST))


def grupo_de_la_semilla(etiquetas, semilla):
    """Máscara con todos los píxeles del mismo grupo k-means que la semilla (estén donde estén)."""
    x, y = semilla
    return np.where(etiquetas == etiquetas[y, x], 255, 0).astype(np.uint8)


def crecer_region(frame, semilla, tolerancia=30):
    """Crecimiento de regiones con floodFill: desde la semilla, agrega vecinos (4-conectados) cuyo color
    difiere a lo más 'tolerancia' del color de la semilla en cada canal."""
    alto, ancho = frame.shape[:2]
    suave = cv2.GaussianBlur(frame, (5, 5), 0)                # sin suavizar, el ruido corta la región
    mascara = np.zeros((alto + 2, ancho + 2), np.uint8)       # floodFill pide 1 píxel extra por lado
    banderas = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    # 4 = vecindad 4 · MASK_ONLY = no pinta la imagen · FIXED_RANGE = compara con la SEMILLA · 255 = valor
    tol = (tolerancia,) * 3
    cv2.floodFill(suave, mascara, semilla, 0, tol, tol, banderas)
    return mascara[1:-1, 1:-1]                                # quitar el borde extra


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    captura, es_archivo, espera = abrir_fuente()

    VENTANA = "Paso 3 - Agrupación"
    cv2.namedWindow(VENTANA)
    barra("K", VENTANA, 4, 8)                     # K de 1 a 8 (4 por defecto)
    barra("Tolerancia", VENTANA, 30, 160)

    # El mosaico muestra cada panel a la mitad del tamaño real; el alto se adapta a la fuente
    estado = {"semilla": None, "celda": (320, 180), "frame": None}

    def al_hacer_clic(evento, x, y, banderas, parametro):
        if evento == cv2.EVENT_LBUTTONDOWN:
            ancho_c, alto_c = estado["celda"]                       # tamaño de un panel del mosaico
            estado["semilla"] = ((x % ancho_c) * 2, (y % alto_c) * 2)   # panel a mitad de tamaño
            print(f"semilla en {estado['semilla']}")

    cv2.setMouseCallback(VENTANA, al_hacer_clic)
    print("Haz clic sobre un objeto en el panel 'Original'.")
    print("Mueve las barras. Esc o Q = salir")

    ultimo = None
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        estado["frame"] = frame
        estado["celda"] = (320, int(frame.shape[0] * 320 / frame.shape[1]))   # alto del panel en el mosaico

        k = max(1, valor("K", VENTANA))
        tolerancia = valor("Tolerancia", VENTANA)

        t0 = time.perf_counter()
        etiquetas, pintada = segmentar_kmeans(frame, k, escala=0.25)
        ms = (time.perf_counter() - t0) * 1000

        if estado["semilla"] is not None:
            grupo = grupo_de_la_semilla(etiquetas, estado["semilla"])
            region = crecer_region(frame, estado["semilla"], tolerancia)
        else:
            grupo = np.zeros_like(frame[:, :, 0])
            region = np.zeros_like(frame[:, :, 0])

        original = frame.copy()
        if estado["semilla"] is not None:
            cv2.drawMarker(original, estado["semilla"], (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

        panel = mosaico(
            [original, pintada, grupo, region],
            ["Original · haz clic para la semilla", f"k-means · K={k} · {ms:.1f} ms",
             f"Grupo de la semilla · {cv2.countNonZero(grupo)} px",
             f"Región crecida · {cv2.countNonZero(region)} px"],
            columnas=2, ancho=320)
        cv2.imshow(VENTANA, panel)

        ultimo = (k, ms, cv2.countNonZero(grupo), cv2.countNonZero(region))
        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break

    if ultimo is not None:
        k, ms, px_grupo, px_region = ultimo
        print("\n== Último frame ==")
        print(f"K = {k}  k-means tardó {ms:.1f} ms")
        print(f"grupo de la semilla : {px_grupo} píxeles")
        print(f"región crecida      : {px_region} píxeles")

    captura.release()
    cv2.destroyAllWindows()
