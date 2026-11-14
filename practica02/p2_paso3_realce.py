"""Práctica 2 · Paso 3 · Realce: contraste y nitidez
====================================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Recuperar el contraste de una imagen mal iluminada con ecualización global
y CLAHE, y realzar los bordes con un kernel de nitidez y unsharp mask.

Ejecutar
--------
    python p2_paso3_realce.py
    python p2_paso3_realce.py --video
    python p2_paso3_realce.py --video --guardar
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

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
def histograma(gris, alto=150):
    """Dibuja el histograma de una imagen gris como una imagen de 256 x alto píxeles."""
    h = cv2.calcHist([gris], [0], None, [256], [0, 256]).ravel()   # cuántos píxeles tienen cada valor 0..255
    h = h / (h.max() + 1e-9) * (alto - 5)                          # escalar al alto del dibujo
    lienzo = np.full((alto, 256), 255, np.uint8)                   # fondo blanco
    for x, y in enumerate(h):
        cv2.line(lienzo, (x, alto), (x, alto - int(y)), 0, 1)      # una barra vertical por nivel de gris
    return lienzo


KERNEL_NITIDEZ = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], np.float32)


def realzar(gris, alfa, beta, clip, fuerza):
    """Devuelve apagada, ecualizada, CLAHE, kernel de nitidez y unsharp mask."""
    apagada = cv2.convertScaleAbs(gris, alpha=alfa, beta=beta)     # mala iluminación: alfa·gris + beta
    ecualizada = cv2.equalizeHist(apagada)                         # ecualización global
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))   # ecualización local con límite
    local = clahe.apply(apagada)
    con_kernel = cv2.filter2D(gris, -1, KERNEL_NITIDEZ)            # nitidez = kernel 3x3
    desenfocada = cv2.GaussianBlur(gris, (0, 0), 3)
    unsharp = cv2.addWeighted(gris, 1 + fuerza, desenfocada, -fuerza, 0)
    return apagada, ecualizada, local, con_kernel, unsharp


def celda(img, titulo, ancho=400, alto=225):
    """Prepara una imagen para el mosaico: 3 canales, tamaño fijo y título."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (ancho, alto))
    escribir(img, titulo, color=(0, 255, 255))
    return img


def panel_contraste(gris, apagada, ecualizada, local):
    """Arriba: las tres versiones; abajo: sus histogramas."""
    arriba = np.hstack([celda(apagada, "Apagada"),
                        celda(ecualizada, "Ecualizada global"),
                        celda(local, "CLAHE")])
    abajo = np.hstack([celda(histograma(apagada), "Histograma apagada"),
                       celda(histograma(ecualizada), "Histograma ecualizada"),
                       celda(histograma(local), "Histograma CLAHE")])
    return np.vstack([arriba, abajo])


def panel_nitidez(original, con_kernel, unsharp):
    """Original, kernel de nitidez y unsharp mask, lado a lado."""
    return np.hstack([celda(original, "Original"),
                      celda(con_kernel, "Kernel de nitidez 3x3"),
                      celda(unsharp, "Unsharp mask")])


def metricas(nombre, img):
    """Imprime min, max y desviación estándar de una imagen gris."""
    print(f"{nombre:<12} min={img.min():>3}  max={img.max():>3}  "
          f"desviación estándar={img.std():5.1f}")


# =====================================================================================
# Modos de ejecución
# =====================================================================================
def modo_interactivo():
    """Dos ventanas en vivo: contraste (con histogramas) y nitidez."""
    captura, es_archivo, espera = abrir_fuente()
    VENTANA = "Paso 3 - Realce"
    cv2.namedWindow(VENTANA)
    barra("Contraste %", VENTANA, 35, 100)
    barra("Brillo", VENTANA, 40, 100)
    barra("Clip CLAHE", VENTANA, 40, 100)          # 40 -> clipLimit = 4.0
    barra("Afilado", VENTANA, 15, 50)              # 15 -> fuerza = 1.5

    ultimo = None
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alfa = valor("Contraste %", VENTANA) / 100
        beta = valor("Brillo", VENTANA)
        clip = valor("Clip CLAHE", VENTANA) / 10
        fuerza = valor("Afilado", VENTANA) / 10

        apagada, ecualizada, local, con_kernel, unsharp = realzar(gris, alfa, beta, clip, fuerza)
        cv2.imshow(VENTANA, np.vstack([panel_contraste(gris, apagada, ecualizada, local),
                                       panel_nitidez(gris, con_kernel, unsharp)]))
        ultimo = (gris, apagada, ecualizada, local)

        if (cv2.waitKey(espera) & 0xFF) in (27, ord("q")):
            break

    captura.release()
    cv2.destroyAllWindows()
    if ultimo:
        print("== Último frame ==")
        for nombre, img in zip(("original", "apagada", "ecualizada", "CLAHE"), ultimo):
            metricas(nombre, img)


def modo_guardar():
    """Genera las capturas de contraste y nitidez de las capturas para el frame 160."""
    captura, es_archivo, espera = abrir_fuente()
    frame = None
    for _ in range(161):
        frame = leer_frame(captura, es_archivo)
    captura.release()
    if frame is None:
        sys.exit("No se pudo leer el frame de trabajo")

    CARPETA_CAPTURAS.mkdir(exist_ok=True)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    alfa, beta, clip, fuerza = 0.35, 40, 4.0, 1.5
    apagada, ecualizada, local, con_kernel, unsharp = realzar(gris, alfa, beta, clip, fuerza)

    print("== Último frame ==")
    for nombre, img in zip(("original", "apagada", "ecualizada", "CLAHE"), (gris, apagada, ecualizada, local)):
        metricas(nombre, img)

    # Ventanas de OpenCV (mosaicos) tal como se ven en vivo
    cv2.imwrite(str(CARPETA_CAPTURAS / "p2_paso3_contraste.png"),
                panel_contraste(gris, apagada, ecualizada, local))
    cv2.imwrite(str(CARPETA_CAPTURAS / "p2_paso3_nitidez.png"),
                panel_nitidez(gris, con_kernel, unsharp))

    # Figura de detalle con el original incluido y ejes en los histogramas
    fig, ejes = plt.subplots(2, 4, figsize=(18, 8), layout="constrained")
    fig.suptitle("Paso 3 · Contraste: original, apagada, ecualización global y CLAHE",
                 fontsize=14, fontweight="bold")
    for col, (titulo, img) in enumerate((("Original", gris), ("Apagada (α=0.35, β=40)", apagada),
                                         ("Ecualizada global", ecualizada), ("CLAHE (clip=4)", local))):
        ejes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
        ejes[0, col].set_title(titulo)
        ejes[0, col].axis("off")
        ejes[1, col].hist(img.ravel(), bins=256, range=(0, 256), color="#333333")
        ejes[1, col].set_xlim(0, 255)
        ejes[1, col].set_yticks([])
        ejes[1, col].set_title(f"Histograma · desv={img.std():.1f}")
    fig.savefig(CARPETA_CAPTURAS / "p2_paso3_contraste_detalle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figura de nitidez: incluye un afilado fuerte para mostrar los halos
    _, _, _, con_kernel5, unsharp5 = realzar(gris, alfa, beta, clip, 5.0)
    fig, ejes = plt.subplots(1, 4, figsize=(18, 5), layout="constrained")
    fig.suptitle("Paso 3 · Nitidez: kernel 3×3, unsharp mask y afilado fuerte (halos)",
                 fontsize=14, fontweight="bold")
    for eje, (titulo, img) in zip(ejes, (("Original", gris), ("Kernel de nitidez", con_kernel),
                                         ("Unsharp mask (f=1.5)", unsharp),
                                         ("Unsharp mask (f=5.0)", unsharp5))):
        eje.imshow(img, cmap="gray", vmin=0, vmax=255)
        eje.set_title(titulo)
        eje.axis("off")
    fig.savefig(CARPETA_CAPTURAS / "p2_paso3_nitidez_detalle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n[capturas] figuras guardadas en reporte/p2_paso3_*.png")


if __name__ == "__main__":
    if MODO_GUARDAR:
        modo_guardar()
    else:
        modo_interactivo()
