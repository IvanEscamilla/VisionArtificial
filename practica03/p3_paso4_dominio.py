"""Práctica 3 · Paso 4 · Conocimiento del dominio
=================================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Quedarse solo con el objeto buscado usando lo que sabemos de él:
color (HSV), tamaño (área) y forma (circularidad).

Ejecutar
--------
    python p3_paso4_dominio.py
    python p3_paso4_dominio.py --video
    python p3_paso4_dominio.py --video --guardar
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from p3_paso1_umbral import limpiar                     # apertura + cierre del paso 1


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
# Funciones nuevas del paso 4
# =====================================================================================
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

    Devuelve (máscara de color, máscara final, aceptados, rechazados).
    """
    color = limpiar(mascara_color(frame, *rango), k)
    contornos, _ = cv2.findContours(color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aceptados, rechazados = [], []
    for c in contornos:
        area, circ, caja = describir(c)
        (aceptados if area >= area_min and circ >= circ_min else rechazados).append((c, area, circ, caja))
    final = np.zeros_like(color)
    cv2.drawContours(final, [a[0] for a in aceptados], -1, 255, cv2.FILLED)
    return color, final, aceptados, rechazados


def anotar(frame, aceptados, rechazados):
    """Dibuja aceptados en verde y rechazados en rojo, con sus medidas."""
    salida = frame.copy()
    for c, area, circ, (x, y, w, h) in aceptados:
        cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(salida, f"OK {area:.0f} c={circ:.2f}", (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    for c, area, circ, (x, y, w, h) in rechazados:
        cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(salida, f"NO {area:.0f} c={circ:.2f}", (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    escribir(salida, f"aceptados: {len(aceptados)}  rechazados: {len(rechazados)}")
    return salida


def muestra_hsv(frame, semilla):
    """Devuelve H, S, V del píxel indicado (para el clic)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = (int(x) for x in hsv[semilla[1], semilla[0]])
    return h, s, v


# =====================================================================================
# Modos de ejecución
# =====================================================================================
def modo_interactivo():
    """Barras de rango HSV, área y circularidad; clic para tomar el color de un objeto."""
    captura, es_archivo, espera = abrir_fuente()
    VENTANA = "Paso 4 - Dominio"
    VENTANA_MASCARAS = "Paso 4 - Mascaras"
    cv2.namedWindow(VENTANA)
    cv2.namedWindow(VENTANA_MASCARAS)
    barra("H min", VENTANA, 5, 179)
    barra("H max", VENTANA, 25, 179)
    barra("S min", VENTANA, 120, 255)
    barra("V min", VENTANA, 80, 255)
    barra("Area min", VENTANA, 1500 // 10, 1000)       # se multiplica por 10
    barra("Circ x100", VENTANA, 90, 100)

    def al_hacer_clic(evento, x, y, banderas, parametro):
        if evento == cv2.EVENT_LBUTTONDOWN:
            frame = estado.get("frame")
            if frame is None:
                return
            h, s, v = muestra_hsv(frame, (x, y))
            cv2.setTrackbarPos("H min", VENTANA, max(0, h - 10))
            cv2.setTrackbarPos("H max", VENTANA, min(179, h + 10))
            cv2.setTrackbarPos("S min", VENTANA, max(0, s - 40))
            cv2.setTrackbarPos("V min", VENTANA, max(0, v - 40))
            print(f"clic en ({x}, {y}): H={h} S={s} V={v}")

    estado = {"frame": None}
    cv2.setMouseCallback(VENTANA, al_hacer_clic)
    print("Haz clic sobre el objeto que quieras segmentar.")

    ultimo = None
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        estado["frame"] = frame
        rango = (valor("H min", VENTANA), valor("H max", VENTANA),
                 valor("S min", VENTANA), valor("V min", VENTANA))
        area_min = valor("Area min", VENTANA) * 10
        circ_min = valor("Circ x100", VENTANA) / 100

        color, final, aceptados, rechazados = segmentar_dominio(frame, rango, area_min, circ_min, 5)
        anotada = anotar(frame, aceptados, rechazados)
        cv2.imshow(VENTANA, anotada)
        cv2.imshow(VENTANA_MASCARAS, mosaico([color, final],
                                             [f"Máscara de color {rango}",
                                              f"Máscara final · {len(aceptados)} aceptados"],
                                             columnas=2, ancho=400))
        ultimo = (rango, area_min, circ_min, aceptados, rechazados)

        if (cv2.waitKey(espera) & 0xFF) in (27, ord("q")):
            break

    captura.release()
    cv2.destroyAllWindows()
    if ultimo:
        rango, area_min, circ_min, aceptados, rechazados = ultimo
        print("== Último frame ==")
        print(f"rango HSV={rango}  área_min={area_min}  circ_min={circ_min:.2f}")
        print(f"{'estado':<10} {'área':>7} {'circularidad':>13}")
        for estado_, lista in (("aceptado", aceptados), ("rechazado", rechazados)):
            for _, area, circ, _ in lista:
                print(f"{estado_:<10} {area:>7.0f} {circ:>13.2f}")


def analizar_overlap():
    """Analiza los frames en los que la pelota se traslapa con el cuadrado (para las capturas)."""
    captura = cv2.VideoCapture(str(RUTA_VIDEO))
    filas = []
    for n in range(216):
        ok, frame = captura.read()
        if not ok:
            break
        if n < 160:
            continue
        color, final, aceptados, rechazados = segmentar_dominio(frame, (5, 25, 120, 80), 1500, 0.90, 5)
        mayor = max(aceptados + rechazados, key=lambda t: t[1], default=None)
        if mayor is not None:
            _, area, circ, _ = mayor
            estado = "aceptado" if any(mayor is a for a in aceptados) else "rechazado"
            filas.append((n, estado, area, circ, len(aceptados), len(rechazados), frame.copy()))
    captura.release()

    with open(CARPETA_CAPTURAS / "p3_paso4_overlap.csv", "w", encoding="utf-8") as f:
        f.write("frame,estado_mayor,area_mayor,circularidad,aceptados,rechazados\n")
        for n, estado, area, circ, na, nr, _ in filas:
            f.write(f"{n},{estado},{area:.1f},{circ:.3f},{na},{nr}\n")

    interesantes = [f for f in filas if f[0] in (165, 175, 185, 195, 205)]
    if interesantes:
        cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso4_overlap.png"),
                    mosaico([anotar(frame, *segmentar_dominio(frame, (5, 25, 120, 80), 1500, 0.90, 5)[2:])
                             for *_, frame in interesantes],
                            [f"frame {n} · {estado}" for n, estado, *_ in interesantes],
                            columnas=2, ancho=420))
    print("\n[capturas] tabla de traslape guardada en reporte/p3_paso4_overlap.csv")


def modo_guardar():
    """Genera las capturas, la tabla y el análisis de traslape de las capturas (frame 160)."""
    captura, es_archivo, espera = abrir_fuente()
    frame = None
    for _ in range(161):
        frame = leer_frame(captura, es_archivo)
    captura.release()
    if frame is None:
        sys.exit("No se pudo leer el frame de trabajo")

    CARPETA_CAPTURAS.mkdir(exist_ok=True)
    rango = (5, 25, 120, 80)
    color, final, aceptados, rechazados = segmentar_dominio(frame, rango, 1500, 0.90, 5)

    print("== Último frame ==")
    print(f"rango HSV={rango}  área_min=1500  circ_min=0.90")
    print(f"{'estado':<10} {'área':>7} {'circularidad':>13}")
    for estado, lista in (("aceptado", aceptados), ("rechazado", rechazados)):
        for _, area, circ, _ in lista:
            print(f"{estado:<10} {area:>7.0f} {circ:>13.2f}")

    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso4_anotada.png"), anotar(frame, aceptados, rechazados))
    cv2.imwrite(str(CARPETA_CAPTURAS / "p3_paso4_mascaras.png"),
                mosaico([color, final],
                        ["Máscara de color (todo lo naranja)", "Máscara final (color + tamaño + forma)"],
                        columnas=2, ancho=520))

    # Valores HSV de los objetos de la escena, para las preguntas
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for nombre, (x, y) in (("pelota naranja", (576, 268)), ("cuadrado naranja", (472, 290)),
                           ("círculo azul", (229, 204)), ("círculo verde", (329, 163))):
        h, s, v = (int(a) for a in hsv[y, x])
        print(f"{nombre:<18} ({x:3d},{y:3d})  H={h:3d} S={s:3d} V={v:3d}")

    analizar_overlap()
    print("\n[capturas] capturas guardadas en reporte/p3_paso4_*.png")


if __name__ == "__main__":
    if MODO_GUARDAR:
        modo_guardar()
    else:
        modo_interactivo()
