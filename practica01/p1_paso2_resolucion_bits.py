"""Práctica 1 · Paso 2 · Resolución espacial y profundidad de bits
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado):
    python p1_paso2_resolucion_bits.py            # usa recursos/astronauta.png
    python p1_paso2_resolucion_bits.py --camara   # usa la webcam (ESPACIO captura)
"""
import sys                        # sys.argv (argumentos) y sys.exit (terminar con un mensaje)
from pathlib import Path          # rutas que funcionan igual en Windows, macOS y Linux

import cv2                        # OpenCV: leer, mostrar y procesar imágenes
import numpy as np                # NumPy: una imagen de OpenCV ES un arreglo de NumPy
import matplotlib.pyplot as plt   # Matplotlib: figuras con varios paneles, títulos y números

# =====================================================================================
# [BASE] Configuración y funciones que se reutilizan en los pasos 2 y 3
# =====================================================================================
CARPETA = Path(__file__).resolve().parent                 # carpeta donde está este script
RUTA_IMAGEN = CARPETA / "recursos" / "webcam.png"        # foto tomada con mi webcam


def cargar_imagen(ruta):
    """Lee una imagen del disco. OpenCV la devuelve en orden de canales B, G, R y tipo uint8."""
    imagen = cv2.imread(str(ruta))                        # str(): imread recibe texto, no un objeto Path
    if imagen is None:                                    # imread NO lanza error: devuelve None si falla
        sys.exit(f"No se pudo leer la imagen: {ruta}")    # terminamos con un mensaje claro
    return imagen


def capturar_webcam(indice=0):
    """Abre la cámara y devuelve una foto. ESPACIO = capturar · ESC = cancelar."""
    camara = cv2.VideoCapture(indice)                     # 0 = primera cámara del equipo
    if not camara.isOpened():                             # ¿la cámara existe y está libre?
        sys.exit("No se pudo abrir la cámara (¿la usa otra aplicación?)")
    foto = None                                           # aquí guardaremos la foto capturada
    while True:                                           # un video es una secuencia de imágenes (frames)
        ok, frame = camara.read()                         # ok = True si llegó un frame; frame = imagen BGR
        if not ok:                                        # la cámara dejó de entregar imágenes
            break
        cv2.imshow("Webcam - ESPACIO captura, ESC cancela", frame)   # mostramos el frame actual
        tecla = cv2.waitKey(1) & 0xFF                     # espera 1 ms por una tecla (y refresca la ventana)
        if tecla == 32:                                   # 32 = código ASCII de la barra espaciadora
            foto = frame.copy()                           # copia: el siguiente read() reutiliza la memoria
            break
        if tecla == 27:                                   # 27 = código de la tecla ESC
            break
    camara.release()                                      # liberar la cámara para otras aplicaciones
    cv2.destroyAllWindows()                               # cerrar la ventana del video
    if foto is None:
        sys.exit("No se capturó ninguna foto")
    return foto


def obtener_imagen():
    """Elige la fuente: webcam si se ejecutó con --camara; si no, el archivo RUTA_IMAGEN."""
    if "--camara" in sys.argv:                            # sys.argv = lista de palabras escritas al ejecutar
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
# Funciones nuevas del paso 2
# =====================================================================================
def reducir_resolucion(img, factor):
    """Muestreo: deja 1 píxel por cada bloque de factor x factor.
    Devuelve la imagen pequeña y la misma ampliada al tamaño original para compararlas lado a lado."""
    alto, ancho = img.shape[:2]
    pequena = cv2.resize(img, (ancho // factor, alto // factor),     # OJO: dsize va como (ANCHO, ALTO)
                         interpolation=cv2.INTER_AREA)               # INTER_AREA promedia el bloque (reducir)
    ampliada = cv2.resize(pequena, (ancho, alto),                    # regresamos al tamaño original...
                          interpolation=cv2.INTER_NEAREST)           # ...repitiendo cada píxel: se ven bloques
    return pequena, ampliada


def reducir_bits(img, bits):
    """Cuantización: deja solo 2**bits niveles por canal y los reparte entre 0 y 255 para poder verlos."""
    niveles = 2 ** bits                                    # 8 bits -> 256 niveles; 1 bit -> 2 niveles
    paso = 256 // niveles                                  # cuántos valores originales caen en cada nivel
    indice = img // paso                                   # división entera: nivel de cada píxel (0..niveles-1)
    return (indice * (255 // (niveles - 1))).astype(np.uint8)   # nivel -> intensidad visible (0..255)


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    # a) Imagen de trabajo: color y gris
    print("== Imagen de trabajo ==")
    imagen = obtener_imagen()
    imprimir_info("color", imagen)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imprimir_info("gris", gris)

    # b) Resolución espacial: menos píxeles, misma profundidad de bits
    print("\n== 1. Resolución espacial (muestreo), imagen a color de 8 bits por canal ==")
    print(f"{'factor':>6} {'resolución':>11} {'píxeles':>9} {'memoria':>10}")
    fig1, ejes1 = plt.subplots(2, 3, figsize=(13, 9), layout="constrained")
    fig1.suptitle("Resolución espacial: de 640×480 a 20×15 píxeles", fontsize=14)
    for eje, factor in zip(ejes1.flat, [1, 2, 4, 8, 16, 32]):        # un panel por factor
        pequena, ampliada = reducir_resolucion(imagen, factor)
        pixeles = pequena.shape[0] * pequena.shape[1]
        print(f"{factor:>6} {f'{pequena.shape[1]}x{pequena.shape[0]}':>11} "
              f"{pixeles:>9} {pequena.nbytes / 1024:>7.1f} KB")
        eje.imshow(a_rgb(ampliada))
        eje.set_title(f"factor {factor} · {pequena.shape[1]}x{pequena.shape[0]} px")
        eje.axis("off")

    # c) Profundidad de bits: menos niveles, mismos píxeles
    print("\n== 2. Profundidad de bits (cuantización), imagen en gris ==")
    print(f"{'bits':>4} {'niveles':>7} {'valores distintos':>17} "
          f"{'memoria teórica':>15} {'memoria real (uint8)':>21}")
    fig2, ejes2 = plt.subplots(3, 5, figsize=(16, 9), layout="constrained")
    fig2.suptitle("Profundidad de bits: de 256 a 2 niveles en gris y a color", fontsize=14)
    for col, bits in enumerate([8, 4, 3, 2, 1]):
        g = reducir_bits(gris, bits)
        c = reducir_bits(imagen, bits)
        niveles = 2 ** bits
        distintos = len(np.unique(g))
        teorica = gris.size * bits / 8 / 1024
        print(f"{bits:>4} {niveles:>7} {distintos:>17} {teorica:>11.0f} KB "
              f"{g.nbytes / 1024:>17.0f} KB")

        ejes2[0, col].imshow(g, cmap="gray", vmin=0, vmax=255)
        ejes2[0, col].set_title(f"{bits} bits · {niveles} niveles")
        ejes2[0, col].axis("off")

        ejes2[1, col].hist(g.ravel(), bins=256, range=(0, 256), color="tab:blue")
        ejes2[1, col].set_title(f"Histograma · {bits} bits")
        ejes2[1, col].set_xlim(0, 255)
        ejes2[1, col].set_yticks([])

        ejes2[2, col].imshow(a_rgb(c))
        ejes2[2, col].set_title(f"Color · {bits} bits/canal")
        ejes2[2, col].axis("off")

    # d) Un solo plt.show() para ver las dos figuras
    plt.show()
