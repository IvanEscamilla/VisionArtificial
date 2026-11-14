"""Práctica 1 · Paso 1: cargar, mostrar e inspeccionar la imagen.

Ejecutar (con el entorno virtual activado):
    python p1_paso1_cargar_inspeccionar.py            (usa recursos/astronauta.png)
    python p1_paso1_cargar_inspeccionar.py --camara   (ESPACIO = capturar, ESC = cancelar)
"""
import sys                           # sys.argv (argumentos) y sys.exit (terminar con un mensaje)
from pathlib import Path             # rutas que funcionan igual en Windows, macOS y Linux

import cv2                           # OpenCV: leer, mostrar y procesar imágenes
import numpy as np                   # NumPy: una imagen de OpenCV ES un arreglo de NumPy
import matplotlib.pyplot as plt      # Matplotlib: figuras con varios paneles, títulos y números
from matplotlib.patches import Rectangle

# =====================================================================================
# [BASE] Configuración y funciones que se reutilizan en los pasos 2 y 3
# =====================================================================================
CARPETA = Path(__file__).resolve().parent                # carpeta donde está este script
RUTA_IMAGEN = CARPETA / "recursos" / "astronauta.png"    # imagen de trabajo (puedes cambiarla por una tuya)


def cargar_imagen(ruta):
    """Lee una imagen del disco. OpenCV la devuelve en orden de canales B, G, R y tipo uint8."""
    imagen = cv2.imread(str(ruta))        # str(): imread recibe texto, no un objeto Path
    if imagen is None:                    # imread NO lanza error: devuelve None si falla
        sys.exit(f"No se pudo leer la imagen: {ruta}")   # terminamos con un mensaje claro
    return imagen


def capturar_webcam(indice=0):
    """Abre la cámara y devuelve una foto. ESPACIO = capturar · ESC = cancelar."""
    camara = cv2.VideoCapture(indice)     # 0 = primera cámara del equipo
    if not camara.isOpened():             # ¿la cámara existe y está libre?
        sys.exit("No se pudo abrir la cámara (¿la usa otra aplicación?)")
    foto = None                           # aquí guardaremos la foto capturada
    while True:                           # un video es una secuencia de imágenes (frames)
        ok, frame = camara.read()         # ok = True si llegó un frame; frame = imagen BGR
        if not ok:                        # la cámara dejó de entregar imágenes
            break
        cv2.imshow("Webcam - ESPACIO captura, ESC cancela", frame)  # mostramos el frame actual
        tecla = cv2.waitKey(1) & 0xFF     # espera 1 ms por una tecla (y refresca la ventana)
        if tecla == 32:                   # 32 = código ASCII de la barra espaciadora
            foto = frame.copy()           # copia: el siguiente read() reutiliza la memoria
            break
        if tecla == 27:                   # 27 = código de la tecla ESC
            break
    camara.release()                      # liberar la cámara para otras aplicaciones
    cv2.destroyAllWindows()               # cerrar la ventana del video
    if foto is None:
        sys.exit("No se capturó ninguna foto")
    return foto


def obtener_imagen():
    """Elige la fuente: webcam si se ejecutó con --camara; si no, el archivo RUTA_IMAGEN."""
    if "--camara" in sys.argv:            # sys.argv = lista de palabras escritas al ejecutar
        return capturar_webcam()
    return cargar_imagen(RUTA_IMAGEN)


def imprimir_info(nombre, img):
    """Imprime las propiedades básicas de una imagen (sirve para color, gris o binaria)."""
    alto, ancho = img.shape[:2]                           # shape = (filas, columnas[, canales])
    canales = 1 if img.ndim == 2 else img.shape[2]        # ndim = 2 -> matriz (gris); 3 -> cubo (color)
    print(f"{nombre:<10} shape={str(img.shape):<15} ancho={ancho:<4} alto={alto:<4} "
          f"canales={canales} dtype={img.dtype} min={img.min():>3} max={img.max():>3} "
          f"memoria={img.nbytes / 1024:.0f} KB")


def a_rgb(img):
    """Matplotlib espera R, G, B; OpenCV entrega B, G, R. Las imágenes en gris se dejan igual."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =====================================================================================
# PROGRAMA PRINCIPAL (solo del paso 1)
# =====================================================================================
if __name__ == "__main__":
    # 1) Obtener la imagen, convertirla a gris e imprimir sus propiedades
    print("== 1. Propiedades ==")
    imagen = obtener_imagen()
    imprimir_info("color", imagen)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imprimir_info("gris", gris)

    # 2) Leer el píxel de la fila 100, columna 200, a color y en gris
    print("\n== 3. Píxel en (y=100, x=200) ==")
    y, x = 100, 200
    b, g, r = imagen[y, x]                # img[fila, columna] -> [B, G, R]
    print(f"color -> B={b} G={g} R={r}")
    print(f"gris -> {gris[y, x]}")

    # 3) Pintar un cuadro rojo de 40x40 píxeles sobre una copia (rojo en OpenCV = (0, 0, 255))
    marcada = imagen.copy()               # sin .copy() modificarías también la original
    marcada[y:y + 40, x:x + 40] = (0, 0, 255)

    # 4) Mostrar las tres imágenes con OpenCV
    cv2.imshow("Original (color, 3 canales)", imagen)
    cv2.imshow("Escala de grises (1 canal)", gris)
    cv2.imshow("Copia con cuadro rojo", marcada)
    print("\nPulsa cualquier tecla sobre una ventana de OpenCV para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5) Recortar una región de 6x6 píxeles que empieza en (y0, x0) = (382, 131)
    N = 6
    y0, x0 = 382, 131
    region_color = imagen[y0:y0 + N, x0:x0 + N]
    region_gris = gris[y0:y0 + N, x0:x0 + N]

    print(f"\n== 6. Región {N}x{N} a partir de (y={y0}, x={x0}) ==")
    print("Gris (1 número por píxel):")
    print(region_gris)
    print("Canal B (region_color[:, :, 0]):")
    print(region_color[:, :, 0])
    print("Canal G (region_color[:, :, 1]):")
    print(region_color[:, :, 1])
    print("Canal R (region_color[:, :, 2]):")
    print(region_color[:, :, 2])

    # 6) Figura de Matplotlib con 3 paneles
    fig, ejes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    fig.suptitle("Paso 1 · La imagen digital es una matriz de números")

    # Panel 1: imagen completa con la región marcada en amarillo
    ejes[0].imshow(a_rgb(imagen))
    ejes[0].add_patch(Rectangle((x0 - 0.5, y0 - 0.5), N, N,
                                edgecolor="yellow", facecolor="none", linewidth=2))
    ejes[0].set_title(f"Original {imagen.shape[1]}x{imagen.shape[0]} px · región en amarillo")
    ejes[0].axis("off")

    # extent hace que los ejes muestren las coordenadas reales (columnas x, filas y) de la región
    extent = (x0 - 0.5, x0 + N - 0.5, y0 + N - 0.5, y0 - 0.5)

    # Panel 2: región a color con los 3 valores (B, G, R) de cada píxel
    ejes[1].imshow(a_rgb(region_color), extent=extent)
    for (fila, col), _ in np.ndenumerate(region_gris):
        bb, gg, rr = region_color[fila, col]
        ejes[1].text(x0 + col, y0 + fila, f"B{bb}\nG{gg}\nR{rr}",
                     ha="center", va="center", color="white", fontsize=7)
    ejes[1].set_title("Región a color: 3 valores por píxel (B, G, R)")
    ejes[1].set_xlabel("columna (x)")
    ejes[1].set_ylabel("fila (y)")

    # Panel 3: región en gris con su valor dentro de cada píxel
    ejes[2].imshow(region_gris, cmap="gray", vmin=0, vmax=255, extent=extent)
    for (fila, col), valor in np.ndenumerate(region_gris):
        ejes[2].text(x0 + col, y0 + fila, str(valor), ha="center", va="center",
                     color="white", fontsize=9)
    ejes[2].set_title("Región en gris: 1 valor por píxel (0-255)")
    ejes[2].set_xlabel("columna (x)")
    ejes[2].set_ylabel("fila (y)")

    plt.show()