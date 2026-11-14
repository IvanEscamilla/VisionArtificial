"""Práctica 3 · Paso 5 · Movimiento
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado):
    python p3_paso5_movimiento.py                  # webcam
    python p3_paso5_movimiento.py --video          # recursos/video_sintetico.mp4

Barras: Alfa x1000 · Umbral · Área min · Teclas: Esc o Q = salir
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
# Funciones de detección de movimiento
# =====================================================================================
from p3_paso1_umbral import limpiar, preparar_gris


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


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    captura, es_archivo, espera = abrir_fuente()

    VENTANA = "Paso 5 - Movimiento"
    cv2.namedWindow(VENTANA)
    barra("Alfa x1000", VENTANA, 20, 100)
    barra("Umbral", VENTANA, 25, 100)
    barra("Área min", VENTANA, 300, 2000)

    promedio = FondoPromedio()
    mog2 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

    print("Mueve las barras. Esc o Q = salir")
    anterior = None
    ultimo = None

    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break

        alfa = valor("Alfa x1000", VENTANA) / 1000
        umbral = valor("Umbral", VENTANA)
        area_min = valor("Área min", VENTANA)

        gris = preparar_gris(frame, 1.5)
        if anterior is None:
            anterior = gris.copy()
            continue

        mascara_dif = limpiar(diferencia_frames(anterior, gris, umbral), 5)
        mascara_prom, fondo_prom = promedio.aplicar(gris, alfa, umbral)
        mascara_prom = limpiar(mascara_prom, 5)

        mascara_mog = mog2.apply(gris)
        mascara_mog = limpiar(mascara_mog, 5)
        fondo_mog = mog2.getBackgroundImage()

        cajas = cajas_en_movimiento(mascara_mog, area_min)
        detecciones = frame.copy()
        for x, y, w, h in cajas:
            cv2.rectangle(detecciones, (x, y), (x + w, y + h), (0, 255, 0), 2)

        panel = mosaico(
            [detecciones, fondo_prom, fondo_mog, mascara_dif, mascara_prom, mascara_mog],
            [f"Detecciones MOG2 · {len(cajas)} objeto(s)",
             f"Fondo promedio (alfa={alfa:.3f})", "Fondo de MOG2",
             f"Diferencia de frames · {100 * cv2.countNonZero(mascara_dif) / mascara_dif.size:.2f} %",
             f"Fondo promedio · {100 * cv2.countNonZero(mascara_prom) / mascara_prom.size:.2f} %",
             f"MOG2 · {100 * cv2.countNonZero(mascara_mog) / mascara_mog.size:.2f} %"],
            columnas=3, ancho=400)
        cv2.imshow(VENTANA, panel)

        ultimo = (100 * cv2.countNonZero(mascara_dif) / mascara_dif.size,
                  100 * cv2.countNonZero(mascara_prom) / mascara_prom.size,
                  100 * cv2.countNonZero(mascara_mog) / mascara_mog.size,
                  len(cajas))
        anterior = gris.copy()

        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break

    if ultimo is not None:
        print("\n== Último frame ==")
        print(f"diferencia {ultimo[0]:.2f} % de la imagen en movimiento")
        print(f"promedio     {ultimo[1]:.2f} % de la imagen en movimiento")
        print(f"MOG2         {ultimo[2]:.2f} % de la imagen en movimiento")
        print(f"objetos en movimiento (MOG2): {ultimo[3]}")

    captura.release()
    cv2.destroyAllWindows()
