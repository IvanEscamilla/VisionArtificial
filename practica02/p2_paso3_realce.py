import sys   # sys.argv (argumentos) y sys.exit (terminar con un mensaje)
import time  # time.perf_counter(): reloj de alta precisión para medir FPS y ms
from pathlib import Path  # rutas que funcionan igual en Windows, macOS y Linux

import cv2           # OpenCV: cámara, video, filtros, segmentación y ventanas
import numpy as np   # NumPy: cada frame es un arreglo de NumPy

# =====================================================================================
# [BASE] Video: abrir la fuente, leer frames, medir FPS, escribir texto, mosaico y barras
# =====================================================================================
CARPETA = Path(__file__).resolve().parent                    # carpeta donde está este script
RUTA_VIDEO = CARPETA / "recursos" / "video_sintetico.mp4"    # video de respaldo (sin cámara)
SALIDAS = CARPETA / "salidas"                                # aquí se guardan capturas y grabaciones


def abrir_fuente():
    """Webcam por defecto; con --video usa RUTA_VIDEO (o la ruta que siga a --video).
    Devuelve (captura, es_archivo, espera_ms)."""
    if "--video" in sys.argv:                                # ¿se pidió un archivo de video?
        i = sys.argv.index("--video")                        # posición de "--video" en la lista
        ruta = sys.argv[i + 1] if i + 1 < len(sys.argv) else RUTA_VIDEO  # ruta escrita o la de respaldo
        captura = cv2.VideoCapture(str(ruta))                # abrir el archivo
        es_archivo = True
    else:
        captura = cv2.VideoCapture(0)                        # 0 = primera cámara del equipo
        es_archivo = False
    if not captura.isOpened():                               # no existe, está ocupada o el códec falla
        sys.exit("No se pudo abrir la fuente de video (¿cámara ocupada? prueba con --video)")
    fps = captura.get(cv2.CAP_PROP_FPS) or 30                # algunos drivers reportan 0 -> usamos 30
    espera = max(1, int(1000 / fps)) if es_archivo else 1    # un archivo se reproduce a su velocidad real
    return captura, es_archivo, espera


def leer_frame(captura, es_archivo):
    """Lee el siguiente frame. Un archivo vuelve a empezar al terminar. Devuelve None si no hay imagen."""
    ok, frame = captura.read()                               # ok = True si llegó un frame
    if not ok and es_archivo:                                # fin del archivo: regresamos al frame 0
        captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = captura.read()
    return frame if ok else None


class MedidorFPS:
    """Mide los cuadros por segundo reales (promedio suavizado)."""

    def __init__(self):
        self.anterior = time.perf_counter()                  # instante del frame anterior
        self.fps = 0.0

    def actualizar(self):
        ahora = time.perf_counter()
        dt = ahora - self.anterior                           # segundos entre este frame y el anterior
        self.anterior = ahora
        if dt > 0:
            self.fps = 1 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt  # suaviza los saltos
        return self.fps


def escribir(img, texto, fila=0, color=(255, 255, 255)):
    """Escribe texto en la esquina superior izquierda, sobre un rectángulo negro para que se lea siempre."""
    y = 22 + 24 * fila                                       # cada fila de texto baja 24 px
    (w, h), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # tamaño que ocupará el texto
    cv2.rectangle(img, (4, y - h - 5), (12 + w, y + base + 2), (0, 0, 0), -1)  # -1 = rectángulo relleno
    cv2.putText(img, texto, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def mosaico(imagenes, titulos, columnas=3, ancho=400):
    """Une varias imágenes (color o gris) en una sola, en una rejilla con título en cada celda."""
    celdas = []
    for img, titulo in zip(imagenes, titulos):
        if img.ndim == 2:                                    # gris o máscara -> 3 canales para poder unirla
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alto = int(img.shape[0] * ancho / img.shape[1])      # conservar la proporción
        celda = cv2.resize(img, (ancho, alto))               # todas las celdas del mismo tamaño
        escribir(celda, titulo, color=(0, 255, 255))         # título en amarillo
        celdas.append(celda)
    while len(celdas) % columnas:                            # completar la última fila con celdas negras
        celdas.append(np.zeros_like(celdas[-1]))
    filas = [np.hstack(celdas[i:i + columnas]) for i in range(0, len(celdas), columnas)]  # pegar en horizontal
    return np.vstack(filas)                                  # y luego las filas en vertical


def barra(nombre, ventana, inicial, maximo):
    """Crea una barra deslizante (trackbar) en una ventana que ya existe."""
    cv2.createTrackbar(nombre, ventana, inicial, maximo, lambda v: None)  # lambda: no hacer nada al moverla


def valor(nombre, ventana):
    """Lee el valor actual de una barra deslizante."""
    return cv2.getTrackbarPos(nombre, ventana)


# =====================================================================================
# Paso 3: histograma como imagen
# =====================================================================================
def histograma(gris, alto=150):
    """Dibuja el histograma de una imagen gris como una imagen de 256 x alto píxeles."""
    h = cv2.calcHist([gris], [0], None, [256], [0, 256]).ravel()  # cuántos píxeles tienen cada valor 0..255
    h = h / (h.max() + 1e-9) * (alto - 5)                    # escalar al alto del dibujo
    lienzo = np.full((alto, 256), 255, np.uint8)             # fondo blanco
    for x, y in enumerate(h):
        cv2.line(lienzo, (x, alto), (x, alto - int(y)), 0, 1)  # una barra vertical por nivel de gris
    return lienzo


# =====================================================================================
# PROGRAMA PRINCIPAL · Paso 3
# =====================================================================================
if __name__ == "__main__":
    captura, es_archivo, espera = abrir_fuente()

    # Kernel de nitidez: el centro menos sus 4 vecinos (los pesos suman 1)
    nitidez = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], np.float32)

    # ---- Ventanas y barras --------------------------------------------------------------
    V_CONTRASTE = "Paso 3 - Contraste"
    V_NITIDEZ = "Paso 3 - Nitidez"
    cv2.namedWindow(V_CONTRASTE)                             # la ventana debe existir antes de crear barras
    cv2.namedWindow(V_NITIDEZ)
    barra("Contraste %", V_CONTRASTE, 35, 100)               # alfa = valor / 100
    barra("Brillo", V_CONTRASTE, 40, 100)                    # beta
    barra("Clip CLAHE", V_CONTRASTE, 40, 100)                # clipLimit = valor / 10  (40 -> 4.0)
    barra("Afilado", V_CONTRASTE, 15, 50)                    # fuerza del unsharp = valor / 10  (15 -> 1.5)
    print("Mueve las barras. Esc o Q = salir")

    ultimo = None                                            # imágenes del último frame para las estadísticas

    # ---- Ciclo de video -----------------------------------------------------------------
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break

        # Leer barras
        alfa = max(valor("Contraste %", V_CONTRASTE), 1) / 100   # mínimo 0.01 para no quedar en negro total
        beta = valor("Brillo", V_CONTRASTE)
        clip = max(valor("Clip CLAHE", V_CONTRASTE), 1) / 10     # mínimo 0.1
        fuerza = valor("Afilado", V_CONTRASTE) / 10

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Contraste ------------------------------------------------------------------
        apagada = cv2.convertScaleAbs(gris, alpha=alfa, beta=beta)   # mala iluminación simulada
        ecualizada = cv2.equalizeHist(apagada)                       # global
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)) # por zonas, con límite de contraste
        local = clahe.apply(apagada)

        # ---- Nitidez --------------------------------------------------------------------
        con_kernel = cv2.filter2D(gris, -1, nitidez)                 # kernel 3x3
        desenfocada = cv2.GaussianBlur(gris, (0, 0), 3)
        unsharp = cv2.addWeighted(gris, 1 + fuerza, desenfocada, -fuerza, 0)  # original + f*(orig - desenf)

        # ---- Mostrar: contraste (arriba imágenes, abajo histogramas) --------------------
        imagenes_c = [apagada, ecualizada, local,
                      histograma(apagada), histograma(ecualizada), histograma(local)]
        titulos_c = [f"Apagada (x{alfa:.2f} + {beta})", "Ecualizada", f"CLAHE clip={clip:.1f}",
                     "Histograma apagada", "Histograma ecualizada", "Histograma CLAHE"]
        cv2.imshow(V_CONTRASTE, mosaico(imagenes_c, titulos_c, columnas=3, ancho=400))

        # ---- Mostrar: nitidez -----------------------------------------------------------
        cv2.imshow(V_NITIDEZ, mosaico([gris, con_kernel, unsharp],
                                      ["Gris original", "Kernel de nitidez 3x3", f"Unsharp mask f={fuerza:.1f}"],
                                      columnas=3, ancho=400))

        ultimo = (gris, apagada, ecualizada, local)

        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break

    captura.release()
    cv2.destroyAllWindows()

    # ---- Estadísticas del último frame ---------------------------------------------------
    if ultimo is not None:
        nombres = ["original", "apagada", "ecualizada", "CLAHE"]
        print("\n== Último frame ==")
        for nombre, img in zip(nombres, ultimo):
            print(f"{nombre:<11} min={int(img.min()):>3}  max={int(img.max()):>3}  "
                  f"desviación estándar={img.std():5.1f}")