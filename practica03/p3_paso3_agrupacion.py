"""Práctica 3 · Paso 3 · Agrupación de píxeles
==============================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Segmentar por parecido de color con k-means y por conectividad con
crecimiento de regiones desde un clic.

Ejecutar
--------
    python p3_paso3_agrupacion.py
    python p3_paso3_agrupacion.py --video
    python p3_paso3_agrupacion.py --video --guardar
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
# Funciones nuevas del paso 3
# =====================================================================================
def segmentar_kmeans(frame, k=4, escala=0.25):
    """k-means sobre los colores (B, G, R) de una versión reducida del frame.

    Devuelve (etiquetas, pintada): el grupo de cada píxel y la imagen pintada
    con el color de su grupo.
    """
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
    """Crecimiento de regiones con floodFill desde la semilla.

    Agrega vecinos (4-conectados) cuyo color difiere a lo más 'tolerancia'
    del color de la semilla en cada canal.
    """
    alto, ancho = frame.shape[:2]
    suave = cv2.GaussianBlur(frame, (5, 5), 0)                # sin suavizar, el ruido corta la región
    mascara = np.zeros((alto + 2, ancho + 2), np.uint8)       # floodFill pide 1 píxel extra por lado
    banderas = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    # 4 = vecindad 4 · MASK_ONLY = no pinta la imagen · FIXED_RANGE = compara con la SEMILLA · 255 = valor
    tol = (tolerancia,) * 3
    cv2.floodFill(suave, mascara, semilla, 0, tol, tol, banderas)
    return mascara[1:-1, 1:-1]                                # quitar el borde extra


def color_etiquetas(etiquetas, k):
    """Convierte etiquetas 0..k-1 en una imagen gris visible."""
    if k <= 1:
        return etiquetas.copy()
    return (etiquetas.astype(np.float32) * (255.0 / (k - 1))).astype(np.uint8)


# =====================================================================================
# Modos de ejecución
# =====================================================================================
def modo_interactivo():
    """Ventana con K y Tolerancia; clic para fijar la semilla; mosaico 2×2."""
    captura, es_archivo, espera = abrir_fuente()
    VENTANA = "Paso 3 - Agrupacion"
    cv2.namedWindow(VENTANA)
    barra("K", VENTANA, 4, 8)
    barra("Tolerancia", VENTANA, 30, 160)

    estado = {"semilla": None, "celda": (320, 180)}

    def al_hacer_clic(evento, x, y, banderas, parametro):
        if evento == cv2.EVENT_LBUTTONDOWN:
            ancho_c, alto_c = estado["celda"]                   # tamaño de un panel del mosaico
            estado["semilla"] = ((x % ancho_c) * 2, (y % alto_c) * 2)   # panel a mitad de tamaño
            print(f"semilla en {estado['semilla']}")

    cv2.setMouseCallback(VENTANA, al_hacer_clic)
    print("Haz clic sobre un objeto en el panel 'Original'.")

    ultimo = None
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        k = max(2, valor("K", VENTANA))
        tolerancia = valor("Tolerancia", VENTANA)

        t0 = time.perf_counter()
        etiquetas, pintada = segmentar_kmeans(frame, k)
        ms = (time.perf_counter() - t0) * 1000

        region = group = None
        if estado["semilla"] is not None:
            group = grupo_de_la_semilla(etiquetas, estado["semilla"])
            region = crecer_region(frame, estado["semilla"], tolerancia)

        original = frame.copy()
        if estado["semilla"] is not None:
            cv2.drawMarker(original, estado["semilla"], (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

        paneles = [original, color_etiquetas(etiquetas, k), pintada,
                   region if region is not None else np.zeros_like(etiquetas)]
        titulos = [
            "Original · clic para semilla",
            f"Etiquetas k-means K={k}",
            f"Pintada · {ms:.2f} ms",
            f"Región crecida · tol={tolerancia}",
        ]
        if group is not None:
            titulos[2] += f" · grupo {int((group > 0).sum())} px"
            titulos[3] += f" · {int((region > 0).sum())} px"
        cv2.imshow(VENTANA, mosaico(paneles, titulos, columnas=2, ancho=320))
        ultimo = (k, ms, group, region)

        if (cv2.waitKey(espera) & 0xFF) in (27, ord("q")):
            break

    captura.release()
    cv2.destroyAllWindows()
    if ultimo:
        k, ms, group, region = ultimo
        print("== Último frame ==")
        print(f"K = {k}   k-means tardó {ms:.2f} ms")
        if group is not None:
            print(f"grupo de la semilla : {int((group > 0).sum())} píxeles")
            print(f"región crecida      : {int((region > 0).sum())} píxeles")


def modo_guardar():
    """Genera el mosaico, el barrido de K y el de tolerancia de las capturas (frame 160)."""
    captura, es_archivo, espera = abrir_fuente()
    frame = None
    for _ in range(161):
        frame = leer_frame(captura, es_archivo)
    captura.release()
    if frame is None:
        sys.exit("No se pudo leer el frame de trabajo")

    CARPETA_CAPTURAS.mkdir(exist_ok=True)
    semilla = (576, 268)                                      # centro de la pelota naranja en el frame 160
    k = 4
    tolerancia = 30

    t0 = time.perf_counter()
    etiquetas, pintada = segmentar_kmeans(frame, k)
    ms = (time.perf_counter() - t0) * 1000
    group = grupo_de_la_semilla(etiquetas, semilla)
    region = crecer_region(frame, semilla, tolerancia)

    print("Haz clic sobre un objeto en el panel 'Original'.")
    print(f"semilla en {semilla}")
    print("== Último frame ==")
    print(f"K = {k}   k-means tardó {ms:.2f} ms")
    print(f"grupo de la semilla : {int((group > 0).sum())} píxeles")
    print(f"región crecida      : {int((region > 0).sum())} píxeles")

    original = frame.copy()
    cv2.drawMarker(original, semilla, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
    titulos = [
        "Original · semilla",
        f"Etiquetas k-means K={k}",
        f"Pintada · {ms:.2f} ms",
        f"Región crecida · tol={tolerancia}\n{int((region > 0).sum())} px",
    ]
    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso3_mosaico.png"),
                mosaico([original, color_etiquetas(etiquetas, k), pintada, region],
                        titulos, columnas=2, ancho=420))

    # Barrido de K: ¿en qué grupo cae el fondo, las líneas y cada objeto?
    paneles_k, titulos_k = [], []
    for kk in (2, 3, 4, 6):
        et, pint = segmentar_kmeans(frame, kk)
        g = grupo_de_la_semilla(et, semilla)
        paneles_k.append(pint)
        titulos_k.append(f"K={kk} · grupo semilla {int((g > 0).sum())} px")
        print(f"K={kk}: grupo de la semilla = {int((g > 0).sum())} px")
    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso3_k.png"),
                mosaico(paneles_k, titulos_k, columnas=2, ancho=420))

    # Barrido de tolerancia: cuándo se derrama al fondo
    paneles_t, titulos_t = [], []
    for tol in (5, 30, 80, 160):
        reg = crecer_region(frame, semilla, tol)
        paneles_t.append(reg)
        titulos_t.append(f"Tolerancia={tol} · {int((reg > 0).sum())} px")
        print(f"Tolerancia={tol}: región = {int((reg > 0).sum())} px")
    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso3_tolerancia.png"),
                mosaico(paneles_t, titulos_t, columnas=2, ancho=420))
    print("\n[capturas] mosaicos guardados en reporte/p3_paso3_*.png")


if __name__ == "__main__":
    if MODO_GUARDAR:
        modo_guardar()
    else:
        modo_interactivo()
