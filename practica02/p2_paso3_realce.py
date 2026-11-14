"""Práctica 2 · Paso 3 · Realce: contraste y nitidez
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado):
    python p2_paso3_realce.py                  # webcam
    python p2_paso3_realce.py --video          # recursos/video_sintetico.mp4

Barras: Contraste % · Brillo · Clip CLAHE · Afilado
Teclas: Esc o Q = salir
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
# Funciones nuevas del paso 3
# =====================================================================================
def histograma(gris, alto=150):
    """Dibuja el histograma de una imagen gris como una imagen de 256 x alto píxeles."""
    h = cv2.calcHist([gris], [0], None, [256], [0, 256]).ravel()     # cuántos píxeles tienen cada valor 0..255
    h = h / (h.max() + 1e-9) * (alto - 5)                            # escalar al alto del dibujo
    lienzo = np.full((alto, 256), 255, np.uint8)                     # fondo blanco
    for x, y in enumerate(h):
        cv2.line(lienzo, (x, alto), (x, alto - int(y)), 0, 1)        # una barra vertical por nivel de gris
    return lienzo


def panel_histograma(gris):
    """Histograma con la proporción del video para que encaje en el mosaico."""
    return cv2.resize(histograma(gris), (640, 360), interpolation=cv2.INTER_NEAREST)


def estadisticas(nombre, img):
    """Imprime mínimo, máximo y desviación estándar de una imagen gris."""
    print(f"{nombre:<11} min={img.min():>3} max={img.max():>3} "
          f"desviación estándar={img.std():>5.1f}")


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    captura, es_archivo, espera = abrir_fuente()

    VENTANA_C = "Paso 3 - Contraste"
    VENTANA_N = "Paso 3 - Nitidez"
    cv2.namedWindow(VENTANA_C)
    cv2.namedWindow(VENTANA_N)
    barra("Contraste %", VENTANA_C, 35, 100)      # alfa x100
    barra("Brillo", VENTANA_C, 40, 150)           # beta
    barra("Clip CLAHE", VENTANA_C, 40, 100)       # clipLimit x10
    barra("Afilado", VENTANA_N, 15, 50)           # fuerza x10

    print("Mueve las barras. Esc o Q = salir")
    ultimo = None

    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alfa = valor("Contraste %", VENTANA_C) / 100
        beta = valor("Brillo", VENTANA_C)
        clip = max(1.0, valor("Clip CLAHE", VENTANA_C) / 10)
        fuerza = valor("Afilado", VENTANA_N) / 10

        # Contraste: cámara mal iluminada -> ecualización global -> CLAHE
        apagada = cv2.convertScaleAbs(gris, alpha=alfa, beta=beta)   # mala iluminación
        ecualizada = cv2.equalizeHist(apagada)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        local = clahe.apply(apagada)

        # Nitidez: kernel de enfoque y unsharp mask
        nitidez = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        con_kernel = cv2.filter2D(gris, -1, nitidez)                 # nitidez = el kernel 3x3
        desenfocada = cv2.GaussianBlur(gris, (0, 0), 3)
        unsharp = cv2.addWeighted(gris, 1 + fuerza, desenfocada, -fuerza, 0)

        panel_c = mosaico(
            [apagada, ecualizada, local,
             panel_histograma(apagada), panel_histograma(ecualizada), panel_histograma(local)],
            ["Apagada (alfa x beta)", "Ecualizada (global)", f"CLAHE (clip {clip:.1f})",
             "Histograma apagada", "Histograma ecualizada", "Histograma CLAHE"],
            columnas=3, ancho=400)
        panel_n = mosaico(
            [gris, con_kernel, unsharp],
            ["Original", "Kernel de nitidez 3x3", f"Unsharp mask (fuerza {fuerza:.1f})"],
            columnas=3, ancho=400)

        cv2.imshow(VENTANA_C, panel_c)
        cv2.imshow(VENTANA_N, panel_n)

        ultimo = (gris, apagada, ecualizada, local)
        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break

    if ultimo is not None:
        print("\n== Último frame ==")
        for nombre, img in zip(["original", "apagada", "ecualizada", "CLAHE"], ultimo):
            estadisticas(nombre, img)

    captura.release()
    cv2.destroyAllWindows()
