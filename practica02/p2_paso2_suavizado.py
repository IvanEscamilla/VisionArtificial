"""Práctica 2 · Paso 2 · Convolución, ruido y suavizado
=======================================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Comparar cuatro filtros de suavizado (promedio, gaussiano, mediana y
bilateral) sobre video en vivo, con ruido gaussiano y de sal y pimienta,
midiendo PSNR y tiempo por frame.

Ejecutar
--------
    python p2_paso2_suavizado.py
    python p2_paso2_suavizado.py --video
    python p2_paso2_suavizado.py --video --guardar
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
# Funciones nuevas del paso 2
# =====================================================================================
def ruido_gaussiano(img, sigma):
    """Suma a cada píxel un número aleatorio con media 0 y desviación sigma (ruido del sensor)."""
    ruido = np.random.normal(0, sigma, img.shape)             # matriz de números aleatorios (float)
    return np.clip(img + ruido, 0, 255).astype(np.uint8)      # clip: recorta a 0..255 antes de volver a uint8


def ruido_sal_pimienta(img, proporcion):
    """Pone en blanco (sal) o negro (pimienta) una proporción de píxeles al azar."""
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


def kernel_promedio(k):
    """Kernel de promedio k x k: todos los pesos valen 1/k² y suman 1."""
    return np.ones((k, k), np.float32) / (k * k)


def aplicar_filtros(frame, ruidoso, k):
    """Aplica los cuatro filtros y devuelve resultados, tiempos y PSNR contra el frame limpio."""
    kernel = kernel_promedio(k)
    promedio, ms_p = medir(cv2.filter2D, ruidoso, -1, kernel)
    gauss, ms_g = medir(cv2.GaussianBlur, ruidoso, (k, k), 0)
    mediana, ms_m = medir(cv2.medianBlur, ruidoso, k)
    bilateral, ms_b = medir(cv2.bilateralFilter, ruidoso, k, 75, 75)
    resultados = {
        "Promedio": (promedio, ms_p, cv2.PSNR(frame, promedio)),
        "Gaussiano": (gauss, ms_g, cv2.PSNR(frame, gauss)),
        "Mediana": (mediana, ms_m, cv2.PSNR(frame, mediana)),
        "Bilateral": (bilateral, ms_b, cv2.PSNR(frame, bilateral)),
    }
    return resultados


def ensuciar(frame, tipo, intensidad):
    """tipo: 0 = sin ruido, 1 = gaussiano, 2 = sal y pimienta.

    Para sal y pimienta, la intensidad 25 de la barra equivale a 5 % de píxeles.
    """
    if tipo == 0:
        return frame.copy(), "sin ruido"
    if tipo == 1:
        return ruido_gaussiano(frame, intensidad), f"gaussiano σ={intensidad}"
    proporcion = intensidad / 500.0
    return ruido_sal_pimienta(frame, proporcion), f"sal y pimienta {100 * proporcion:.1f} %"


# =====================================================================================
# Modos de ejecución
# =====================================================================================
def modo_interactivo():
    """Ventana con barras: Kernel, Ruido e Intensidad; mosaico 2×3 en vivo."""
    captura, es_archivo, espera = abrir_fuente()
    frame = leer_frame(captura, es_archivo)
    if frame is None:
        sys.exit("La fuente no entregó ningún frame")

    # Convolución a mano: kernel promedio 5x5 y comparación con cv2.blur
    k = 5
    kernel = kernel_promedio(k)
    print("== 1. Kernel promedio 5x5 ==")
    print(np.round(kernel, 2))
    a_mano = cv2.filter2D(frame, -1, kernel)
    diferencia = cv2.absdiff(a_mano, cv2.blur(frame, (k, k))).max()
    print("diferencia máxima filter2D vs cv2.blur:", diferencia)
    print("\nMueve las barras. Esc o Q = salir")

    VENTANA = "Paso 2 - Suavizado"
    cv2.namedWindow(VENTANA)
    barra("Kernel", VENTANA, 2, 7)              # k = 2·valor + 1
    barra("Ruido", VENTANA, 1, 2)               # 0 ninguno, 1 gaussiano, 2 sal y pimienta
    barra("Intensidad", VENTANA, 25, 100)       # sigma o porcentaje/5

    ultimo = None
    leidos = 0
    while True:
        k = 2 * valor("Kernel", VENTANA) + 1
        tipo = valor("Ruido", VENTANA)
        intensidad = valor("Intensidad", VENTANA)
        ruidoso, etiqueta_ruido = ensuciar(frame, tipo, intensidad)
        resultados = aplicar_filtros(frame, ruidoso, k)
        ultimo = (etiqueta_ruido, k, resultados, cv2.PSNR(frame, ruidoso))

        titulos = [
            "Original",
            f"Ruidoso: {etiqueta_ruido}\nPSNR {cv2.PSNR(frame, ruidoso):4.1f} dB",
            f"Promedio k={k}\nPSNR {resultados['Promedio'][2]:4.1f} dB · {resultados['Promedio'][1]:.2f} ms",
            f"Gaussiano k={k}\nPSNR {resultados['Gaussiano'][2]:4.1f} dB · {resultados['Gaussiano'][1]:.2f} ms",
            f"Mediana k={k}\nPSNR {resultados['Mediana'][2]:4.1f} dB · {resultados['Mediana'][1]:.2f} ms",
            f"Bilateral k={k}\nPSNR {resultados['Bilateral'][2]:4.1f} dB · {resultados['Bilateral'][1]:.2f} ms",
        ]
        panel = mosaico([frame, ruidoso] + [r[0] for r in resultados.values()],
                        titulos, columnas=3, ancho=400)
        cv2.imshow(VENTANA, panel)

        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        leidos += 1

    captura.release()
    cv2.destroyAllWindows()

    if ultimo is not None:
        etiqueta_ruido, k, resultados, psnr_ruidoso = ultimo
        print(f"\n== 6. Último frame: ruido {etiqueta_ruido}, kernel {k}x{k} ==")
        print(f"{'filtro':<12} {'PSNR (dB)':>10} {'tiempo (ms)':>12}")
        print(f"{'ruidoso':<12} {psnr_ruidoso:>10.1f} {'-':>12}")
        for nombre, (_, ms, psnr) in resultados.items():
            print(f"{nombre:<12} {psnr:>10.1f} {ms:>12.2f}")


def modo_guardar():
    """Genera las tablas de PSNR y los mosaicos de las capturas para el frame 160."""
    captura, es_archivo, espera = abrir_fuente()
    frame = None
    for _ in range(161):
        frame = leer_frame(captura, es_archivo)
    captura.release()
    if frame is None:
        sys.exit("No se pudo leer el frame de trabajo")

    CARPETA_CAPTURAS.mkdir(exist_ok=True)
    np.random.seed(7)                                        # resultados reproducibles en las capturas

    print("== 1. Kernel promedio 5x5 ==")
    k = 5
    kernel = kernel_promedio(k)
    print(np.round(kernel, 2))
    print("diferencia máxima filter2D vs cv2.blur:",
          cv2.absdiff(cv2.filter2D(frame, -1, kernel), cv2.blur(frame, (k, k))).max())

    # Tabla completa: dos tipos de ruido y varios tamaños de kernel
    filas = []
    for nombre_ruido, tipo, intensidad in (("gaussiano", 1, 25), ("sal y pimienta", 2, 25)):
        for k in (3, 5, 7, 9, 11, 13, 15):
            ruidoso, etiqueta = ensuciar(frame, tipo, intensidad)
            res = aplicar_filtros(frame, ruidoso, k)
            fila = {"ruido": nombre_ruido, "k": k, "ruidoso_psnr": cv2.PSNR(frame, ruidoso)}
            for nombre, (_, ms, psnr) in res.items():
                fila[nombre] = (psnr, ms)
            filas.append(fila)

    print("\n== 2. PSNR (dB) y tiempo (ms) por filtro ==")
    print(f"{'ruido':<15} {'k':>2} {'ruidoso':>8} {'Promedio':>14} {'Gaussiano':>14} "
          f"{'Mediana':>14} {'Bilateral':>14}")
    for f in filas:
        print(f"{f['ruido']:<15} {f['k']:>2} {f['ruidoso_psnr']:>8.1f} "
              f"{f['Promedio'][0]:>7.1f}/{f['Promedio'][1]:<5.2f} "
              f"{f['Gaussiano'][0]:>7.1f}/{f['Gaussiano'][1]:<5.2f} "
              f"{f['Mediana'][0]:>7.1f}/{f['Mediana'][1]:<5.2f} "
              f"{f['Bilateral'][0]:>7.1f}/{f['Bilateral'][1]:<5.2f}")

    # Mosaicos k=5 para los dos ruidos
    for nombre_ruido, tipo, archivo in (("gaussiano", 1, "p2_paso2_gaussiano.png"),
                                        ("sal y pimienta", 2, "p2_paso2_sal_pimienta.png")):
        ruidoso, etiqueta = ensuciar(frame, tipo, 25)
        res = aplicar_filtros(frame, ruidoso, 5)
        titulos = [
            "Original",
            f"Ruidoso: {etiqueta}\nPSNR {cv2.PSNR(frame, ruidoso):.1f} dB",
            f"Promedio 5x5\nPSNR {res['Promedio'][2]:.1f} dB · {res['Promedio'][1]:.2f} ms",
            f"Gaussiano 5x5\nPSNR {res['Gaussiano'][2]:.1f} dB · {res['Gaussiano'][1]:.2f} ms",
            f"Mediana 5x5\nPSNR {res['Mediana'][2]:.1f} dB · {res['Mediana'][1]:.2f} ms",
            f"Bilateral 5x5\nPSNR {res['Bilateral'][2]:.1f} dB · {res['Bilateral'][1]:.2f} ms",
        ]
        panel = mosaico([frame, ruidoso] + [r[0] for r in res.values()],
                        titulos, columnas=3, ancho=400)
        cv2.imwrite(str(CARPETA_CAPTURAS / archivo), panel)

    # Tabla CSV para las capturas
    with open(CARPETA_CAPTURAS / "p2_psnr.csv", "w", encoding="utf-8") as f:
        f.write("ruido,k,ruidoso,Promedio_PSNR,Promedio_ms,Gaussiano_PSNR,Gaussiano_ms,"
                "Mediana_PSNR,Mediana_ms,Bilateral_PSNR,Bilateral_ms\n")
        for f_ in filas:
            f.write(f"{f_['ruido']},{f_['k']},{f_['ruidoso_psnr']:.3f},"
                    + ",".join(f"{f_[n][0]:.3f},{f_[n][1]:.3f}" for n in
                               ("Promedio", "Gaussiano", "Mediana", "Bilateral")) + "\n")

    print("\n[capturas] mosaicos y tabla guardados en reporte/p2_paso2_*.png y p2_psnr.csv")


if __name__ == "__main__":
    if MODO_GUARDAR:
        modo_guardar()
    else:
        modo_interactivo()
