"""Práctica 3 · Paso 4 · Conocimiento del dominio
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado):
    python p3_paso4_dominio.py                  # webcam
    python p3_paso4_dominio.py --video          # recursos/video_sintetico.mp4

Haz clic sobre el objeto para tomar su color (mueve las barras de H, S y V).
Barras: H min · H max · S min · V min · Área min · Circ x100 · Teclas: Esc o Q = salir
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
# Funciones de segmentación por conocimiento del dominio
# =====================================================================================
from p3_paso1_umbral import limpiar                           # apertura + cierre del paso 1


def mascara_color(frame, h_min, h_max, s_min, v_min):
    """Píxeles cuyo tono está entre h_min y h_max (0..179 en OpenCV) con saturación y brillo suficientes."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, 255, 255))   # 255 dentro del rango, 0 fuera


def describir(contorno):
    """Área, circularidad y rectángulo que encierra al contorno."""
    area = cv2.contourArea(contorno)
    casco = cv2.convexHull(contorno)                          # el borde en píxeles es escalonado y su perímetro
    perimetro = cv2.arcLength(casco, True)                    # sale inflado; el casco convexo lo suaviza
    circularidad = 4 * np.pi * area / perimetro ** 2 if perimetro > 0 else 0
    return area, circularidad, cv2.boundingRect(contorno)     # boundingRect = (x, y, ancho, alto)


def segmentar_dominio(frame, rango=(5, 25, 120, 80), area_min=1500, circ_min=0.90, k=5):
    """Color -> limpieza -> contornos -> filtro por tamaño y forma.
    Devuelve (máscara de color, máscara final, aceptados, rechazados)."""
    color = limpiar(mascara_color(frame, *rango), k)
    contornos, _ = cv2.findContours(color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aceptados, rechazados = [], []
    for c in contornos:
        area, circ, caja = describir(c)
        (aceptados if area >= area_min and circ >= circ_min else rechazados).append((c, area, circ, caja))
    final = np.zeros_like(color)
    cv2.drawContours(final, [a[0] for a in aceptados], -1, 255, cv2.FILLED)
    return color, final, aceptados, rechazados


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    captura, es_archivo, espera = abrir_fuente()

    VENTANA = "Paso 4 - Dominio"
    VENTANA_M = "Paso 4 - Máscaras"
    cv2.namedWindow(VENTANA)
    cv2.namedWindow(VENTANA_M)
    barra("H min", VENTANA, 5, 179)
    barra("H max", VENTANA, 25, 179)
    barra("S min", VENTANA, 120, 255)
    barra("V min", VENTANA, 80, 255)
    barra("Área min", VENTANA, 1500, 8000)
    barra("Circ x100", VENTANA, 90, 100)

    estado = {"frame": None}

    def al_hacer_clic(evento, x, y, banderas, parametro):
        if evento == cv2.EVENT_LBUTTONDOWN and estado["frame"] is not None:
            hsv = cv2.cvtColor(estado["frame"], cv2.COLOR_BGR2HSV)
            h, s, v = (int(val) for val in hsv[y, x])
            print(f"clic en ({x}, {y}): H={h} S={s} V={v}")
            cv2.setTrackbarPos("H min", VENTANA, max(0, h - 10))
            cv2.setTrackbarPos("H max", VENTANA, min(179, h + 10))
            cv2.setTrackbarPos("S min", VENTANA, max(0, s - 60))
            cv2.setTrackbarPos("V min", VENTANA, max(0, v - 60))

    cv2.setMouseCallback(VENTANA, al_hacer_clic)
    print("Haz clic sobre el objeto para tomar su color. Esc o Q = salir")

    ultimo = None
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        estado["frame"] = frame

        rango = (valor("H min", VENTANA), valor("H max", VENTANA),
                 valor("S min", VENTANA), valor("V min", VENTANA))
        area_min = valor("Área min", VENTANA)
        circ_min = valor("Circ x100", VENTANA) / 100

        color, final, aceptados, rechazados = segmentar_dominio(
            frame, rango, area_min=area_min, circ_min=circ_min, k=5)

        # Dibujar aceptados en verde y rechazados en rojo, con sus números
        anotada = frame.copy()
        for c, area, circ, (x, y, w, h) in aceptados:
            cv2.rectangle(anotada, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(anotada, f"OK {area:.0f} {circ:.2f}", (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        for c, area, circ, (x, y, w, h) in rechazados:
            cv2.rectangle(anotada, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(anotada, f"{area:.0f} {circ:.2f}", (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        escribir(anotada, f"aceptados: {len(aceptados)} · rechazados: {len(rechazados)}",
                 color=(0, 255, 0))

        cv2.imshow(VENTANA, anotada)
        cv2.imshow(VENTANA_M, mosaico([color, final],
                                      ["Máscara de color (todo lo naranja)", "Máscara final (color + tamaño + forma)"],
                                      columnas=2, ancho=400))

        ultimo = (aceptados, rechazados)
        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break

    if ultimo is not None:
        aceptados, rechazados = ultimo
        print("\n== Último frame ==")
        print(f"{'estado':<12} {'área':>7} {'circularidad':>13}")
        for _, area, circ, _ in aceptados:
            print(f"{'aceptado':<12} {area:>7.0f} {circ:>13.2f}")
        for _, area, circ, _ in rechazados:
            print(f"{'rechazado':<12} {area:>7.0f} {circ:>13.2f}")

    captura.release()
    cv2.destroyAllWindows()
