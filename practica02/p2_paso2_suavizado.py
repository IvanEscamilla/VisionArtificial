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
# Paso 2: ruido y medición de tiempo
# =====================================================================================
def ruido_gaussiano(img, sigma):
    """Suma a cada píxel un número aleatorio con media 0 y desviación sigma (ruido del sensor)."""
    ruido = np.random.normal(0, sigma, img.shape)            # matriz de números aleatorios (float)
    return np.clip(img + ruido, 0, 255).astype(np.uint8)     # clip: recorta a 0..255 antes de volver a uint8


def ruido_sal_pimienta(img, proporcion):
    """Pone en blanco (sal) o negro (pimienta) una proporción de píxeles al azar (fallas de transmisión)."""
    salida = img.copy()
    azar = np.random.rand(*img.shape[:2])                    # un número 0..1 por píxel
    salida[azar < proporcion / 2] = 0                        # pimienta
    salida[azar > 1 - proporcion / 2] = 255                  # sal
    return salida


def medir(funcion, *argumentos):
    """Ejecuta funcion(*argumentos) y devuelve (resultado, milisegundos que tardó)."""
    t0 = time.perf_counter()
    resultado = funcion(*argumentos)
    return resultado, (time.perf_counter() - t0) * 1000


# =====================================================================================
# PROGRAMA PRINCIPAL · Paso 2
# =====================================================================================
if __name__ == "__main__":
    captura, es_archivo, espera = abrir_fuente()
    medidor = MedidorFPS()

    # ---- 3. Convolución a mano: kernel promedio 5x5 y comparación con cv2.blur ---------
    frame = leer_frame(captura, es_archivo)
    if frame is None:
        sys.exit("No llegó ningún frame de la fuente.")
    k = 5
    kernel = np.ones((k, k), np.float32) / (k * k)           # 25 pesos de 1/25: suman 1
    a_mano = cv2.filter2D(frame, -1, kernel)                 # convolución con nuestro kernel
    print("== 1. Kernel promedio 5x5 ==")
    print(kernel)
    print("diferencia máxima filter2D vs cv2.blur:", cv2.absdiff(a_mano, cv2.blur(frame, (k, k))).max())

    # ---- 4. Ventana y barras ------------------------------------------------------------
    VENTANA = "Paso 2 - Suavizado"
    cv2.namedWindow(VENTANA)                                 # la ventana debe existir antes de crear barras
    barra("Kernel", VENTANA, 2, 7)                           # k = 2*valor + 1  -> 1, 3, 5, ..., 15
    barra("Ruido", VENTANA, 1, 2)                            # 0 ninguno, 1 gaussiano, 2 sal y pimienta
    barra("Intensidad", VENTANA, 25, 100)                    # sigma (gaussiano) o % x5 (sal y pimienta)
    print("\nMueve las barras. Esc o Q = salir")

    ultimo = None                                            # resultados del último frame para la tabla final

    # ---- 5. Ciclo de video --------------------------------------------------------------
    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        fps = medidor.actualizar()

        # Leer las barras
        k = 2 * valor("Kernel", VENTANA) + 1                 # siempre impar: lo exigen Gaussian y median
        tipo = valor("Ruido", VENTANA)
        intensidad = valor("Intensidad", VENTANA)

        # Ensuciar el frame según el tipo de ruido
        if tipo == 1:
            ruidoso = ruido_gaussiano(frame, intensidad)
            descripcion = f"gaussiano sigma={intensidad}"
        elif tipo == 2:
            proporcion = intensidad / 500                    # intensidad 25 -> 5 % de los píxeles
            ruidoso = ruido_sal_pimienta(frame, proporcion)
            descripcion = f"sal y pimienta {proporcion * 100:.0f}%"
        else:
            ruidoso = frame.copy()
            descripcion = "sin ruido"

        # Aplicar los 4 filtros midiendo su tiempo
        kernel = np.ones((k, k), np.float32) / (k * k)
        promedio, ms_p = medir(cv2.filter2D, ruidoso, -1, kernel)
        gauss, ms_g = medir(cv2.GaussianBlur, ruidoso, (k, k), 0)
        mediana, ms_m = medir(cv2.medianBlur, ruidoso, k)
        bilateral, ms_b = medir(cv2.bilateralFilter, ruidoso, k, 75, 75)

        # PSNR de cada versión contra el frame limpio
        psnr_ruido = cv2.PSNR(frame, ruidoso)
        psnr_p = cv2.PSNR(frame, promedio)
        psnr_g = cv2.PSNR(frame, gauss)
        psnr_m = cv2.PSNR(frame, mediana)
        psnr_b = cv2.PSNR(frame, bilateral)

        # ---- 6. Mosaico de 2x3 ----------------------------------------------------------
        original = frame.copy()
        escribir(original, f"{fps:.1f} FPS", fila=1)         # los FPS van en la 2.ª línea del original
        imagenes = [original, ruidoso, promedio, gauss, mediana, bilateral]
        titulos = [
            "Original",
            f"Ruido: {descripcion} PSNR {psnr_ruido:.1f} dB",
            f"Promedio k={k} {psnr_p:.1f} dB {ms_p:.1f} ms",
            f"Gaussiano k={k} {psnr_g:.1f} dB {ms_g:.1f} ms",
            f"Mediana k={k} {psnr_m:.1f} dB {ms_m:.1f} ms",
            f"Bilateral k={k} {psnr_b:.1f} dB {ms_b:.1f} ms",
        ]
        cv2.imshow(VENTANA, mosaico(imagenes, titulos, columnas=3, ancho=400))

        # Guardar datos del último frame para la tabla final
        ultimo = {
            "encabezado": f"ruido {descripcion}, kernel {k}x{k}",
            "filas": [
                ("ruidoso", psnr_ruido, None),
                ("Promedio", psnr_p, ms_p),
                ("Gaussiano", psnr_g, ms_g),
                ("Mediana", psnr_m, ms_m),
                ("Bilateral", psnr_b, ms_b),
            ],
        }

        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break

    captura.release()
    cv2.destroyAllWindows()

    # ---- 7. Tabla final del último frame -------------------------------------------------
    if ultimo is not None:
        print(f"\n== 6. Último frame: {ultimo['encabezado']} ==")
        print(f"{'filtro':<12}{'PSNR (dB)':>10}{'tiempo (ms)':>14}")
        for nombre, psnr, ms in ultimo["filas"]:
            tiempo = "-" if ms is None else f"{ms:.2f}"
            print(f"{nombre:<12}{psnr:>10.1f}{tiempo:>14}")