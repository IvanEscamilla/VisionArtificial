"""Práctica 2 · Paso 2 · Convolución, ruido y suavizado
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado):
    python p2_paso2_suavizado.py                  # webcam
    python p2_paso2_suavizado.py --video          # recursos/video_sintetico.mp4

Barras: Kernel · Ruido (0 ninguno, 1 gaussiano, 2 sal y pimienta) · Intensidad
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
# Funciones nuevas del paso 2
# =====================================================================================
def ruido_gaussiano(img, sigma):
    """Suma a cada píxel un número aleatorio con media 0 y desviación sigma (ruido del sensor)."""
    ruido = np.random.normal(0, sigma, img.shape)             # matriz de números aleatorios (float)
    return np.clip(img + ruido, 0, 255).astype(np.uint8)      # clip: recorta a 0..255 antes de volver a uint8


def ruido_sal_pimienta(img, proporcion):
    """Pone en blanco (sal) o negro (pimienta) una proporción de píxeles al azar (fallas de transmisión)."""
    salida = img.copy()
    azar = np.random.rand(*img.shape[:2])                     # un número 0..1 por píxel
    salida[azar < proporcion / 2] = 0                         # pimienta
    salida[azar > 1 - proporcion / 2] = 255                   # sal
    return salida


def medir(funcion, *argumentos):
    """Ejecuta funcion(*argumentos) y devuelve (resultado, milisegundos que tardó)."""
    t0 = time.perf_counter()
    resultado = funcion(*argumentos)
    return resultado, (time.perf_counter() - t0) * 1000


def psnr_texto(referencia, imagen):
    """PSNR en dB con un decimal. Si son idénticas, OpenCV devuelve un valor enorme (≈361 dB)."""
    return f"{cv2.PSNR(referencia, imagen):5.1f}"


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    captura, es_archivo, espera = abrir_fuente()

    # 3) Convolución a mano: kernel promedio 5x5 contra cv2.blur
    frame0 = leer_frame(captura, es_archivo)
    if frame0 is None:
        sys.exit("No se pudo leer el primer frame")

    k = 5
    kernel = np.ones((k, k), np.float32) / (k * k)
    a_mano = cv2.filter2D(frame0, -1, kernel)
    print("== 1. Kernel promedio 5x5 ==")
    np.set_printoptions(precision=2, suppress=True)
    print(kernel)
    print("diferencia máxima filter2D vs cv2.blur:",
          cv2.absdiff(a_mano, cv2.blur(frame0, (k, k))).max())

    # 4) Ventana y tres barras
    VENTANA = "Paso 2 - Suavizado"
    cv2.namedWindow(VENTANA)                      # la ventana debe existir antes de crear barras
    barra("Kernel", VENTANA, 2, 7)                # k = 2*valor + 1  -> 5 por defecto (índices 3..15)
    barra("Ruido", VENTANA, 1, 2)                 # 0 ninguno · 1 gaussiano · 2 sal y pimienta
    barra("Intensidad", VENTANA, 25, 100)         # sigma del gaussiano o % x5 de sal y pimienta

    print("\nMueve las barras. Esc o Q = salir")
    ultimo = None

    # 5) y 6) Ciclo: ruido, cuatro filtros, PSNR y mosaico
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break

        k = 2 * valor("Kernel", VENTANA) + 1
        kernel = np.ones((k, k), np.float32) / (k * k)
        tipo = valor("Ruido", VENTANA)
        intensidad = valor("Intensidad", VENTANA)

        if tipo == 1:
            ruidoso = ruido_gaussiano(frame, intensidad)
            etiqueta = f"gaussiano σ={intensidad}"
        elif tipo == 2:
            proporcion = intensidad / 500.0             # 25 -> 0.05 (5 % de píxeles alterados)
            ruidoso = ruido_sal_pimienta(frame, proporcion)
            etiqueta = f"sal y pimienta {100 * proporcion:.1f} %"
        else:
            ruidoso = frame.copy()
            etiqueta = "sin ruido"

        promedio, ms_p = medir(cv2.filter2D, ruidoso, -1, kernel)
        gauss, ms_g = medir(cv2.GaussianBlur, ruidoso, (k, k), 0)
        mediana, ms_m = medir(cv2.medianBlur, ruidoso, k)
        bilateral, ms_b = medir(cv2.bilateralFilter, ruidoso, k, 75, 75)

        titulos = [
            "Original",
            f"Ruidoso · {psnr_texto(frame, ruidoso)} dB",
            f"Promedio · {psnr_texto(frame, promedio)} dB · {ms_p:.1f} ms",
            f"Gaussiano · {psnr_texto(frame, gauss)} dB · {ms_g:.1f} ms",
            f"Mediana · {psnr_texto(frame, mediana)} dB · {ms_m:.1f} ms",
            f"Bilateral · {psnr_texto(frame, bilateral)} dB · {ms_b:.1f} ms",
        ]
        panel = mosaico([frame, ruidoso, promedio, gauss, mediana, bilateral],
                        titulos, columnas=3, ancho=400)
        cv2.imshow(VENTANA, panel)

        ultimo = {
            "k": k, "etiqueta": etiqueta,
            "ruidoso": cv2.PSNR(frame, ruidoso),
            "promedio": (cv2.PSNR(frame, promedio), ms_p),
            "gaussiano": (cv2.PSNR(frame, gauss), ms_g),
            "mediana": (cv2.PSNR(frame, mediana), ms_m),
            "bilateral": (cv2.PSNR(frame, bilateral), ms_b),
        }

        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break

    # 7) Tabla final con el último frame
    if ultimo is not None:
        print(f"\n== 6. Último frame: ruido {ultimo['etiqueta']}, kernel {ultimo['k']}x{ultimo['k']} ==")
        print(f"{'filtro':<12} {'PSNR (dB)':>9} {'tiempo (ms)':>12}")
        print(f"{'ruidoso':<12} {ultimo['ruidoso']:>9.1f} {'-':>12}")
        for nombre in ["promedio", "gaussiano", "mediana", "bilateral"]:
            db, ms = ultimo[nombre]
            print(f"{nombre.capitalize():<12} {db:>9.1f} {ms:>12.2f}")

    captura.release()
    cv2.destroyAllWindows()
