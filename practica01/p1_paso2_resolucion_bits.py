"""Práctica 1 · Paso 2: resolución espacial (muestreo) y profundidad de bits (cuantización).

Ejecutar (con el entorno virtual activado):
    python p1_paso2_resolucion_bits.py            (usa recursos/astronauta.png)
    python p1_paso2_resolucion_bits.py --camara   (ESPACIO = capturar, ESC = cancelar)
"""
import sys                           # sys.argv (argumentos) y sys.exit (terminar con un mensaje)
from pathlib import Path             # rutas que funcionan igual en Windows, macOS y Linux

import cv2                           # OpenCV: leer, mostrar y procesar imágenes
import numpy as np                   # NumPy: una imagen de OpenCV ES un arreglo de NumPy
import matplotlib.pyplot as plt      # Matplotlib: figuras con varios paneles, títulos y números

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
# Funciones nuevas del paso 2
# =====================================================================================
def reducir_resolucion(img, factor):
    """Muestreo: deja 1 píxel por cada bloque de factor x factor.
    Devuelve la imagen pequeña y la misma ampliada al tamaño original para compararlas lado a lado."""
    alto, ancho = img.shape[:2]
    pequena = cv2.resize(img, (ancho // factor, alto // factor),    # OJO: dsize va como (ANCHO, ALTO)
                         interpolation=cv2.INTER_AREA)              # INTER_AREA promedia el bloque (reducir)
    ampliada = cv2.resize(pequena, (ancho, alto),                   # regresamos al tamaño original...
                          interpolation=cv2.INTER_NEAREST)          # ...repitiendo cada píxel: se ven bloques
    return pequena, ampliada


def reducir_bits(img, bits):
    """Cuantización: deja solo 2**bits niveles por canal y los reparte entre 0 y 255 para poder verlos."""
    niveles = 2 ** bits                                   # 8 bits -> 256 niveles; 1 bit -> 2 niveles
    paso = 256 // niveles                                 # cuántos valores originales caen en cada nivel
    indice = img // paso                                  # división entera: nivel de cada píxel (0..niveles-1)
    return (indice * (255 // (niveles - 1))).astype(np.uint8)   # nivel -> intensidad visible (0..255)


# =====================================================================================
# PROGRAMA PRINCIPAL (solo del paso 2)
# =====================================================================================
if __name__ == "__main__":
    # a) Obtener la imagen y su versión en gris
    print("== Imagen de trabajo ==")
    imagen = obtener_imagen()
    imprimir_info("color", imagen)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imprimir_info("gris", gris)

    # b) Resolución espacial: factores 1, 2, 4, 8, 16, 32
    print("\n== 1. Resolución espacial (muestreo), imagen a color de 8 bits por canal ==")
    print(f"{'factor':<8}{'resolución':<13}{'píxeles':<10}{'memoria'}")

    fig1, ejes1 = plt.subplots(2, 3, figsize=(13, 9), layout="constrained")   # 2 filas x 3 columnas
    fig1.suptitle("Paso 2a · Resolución espacial: mismos 8 bits por canal, cada vez menos píxeles")

    for eje, factor in zip(ejes1.flat, [1, 2, 4, 8, 16, 32]):
        pequena, ampliada = reducir_resolucion(imagen, factor)
        alto_p, ancho_p = pequena.shape[:2]
        memoria_kb = pequena.nbytes / 1024
        print(f"{factor:<8}{f'{ancho_p}x{alto_p}':<13}{ancho_p * alto_p:<10}{memoria_kb:.1f} KB")
        eje.imshow(a_rgb(ampliada))                       # a color: convertir a RGB
        eje.set_title(f"{ancho_p}x{alto_p} px · {memoria_kb:.1f} KB")
        eje.axis("off")

    # c) Profundidad de bits: 8, 4, 3, 2, 1 con la imagen gris y a color
    print("\n== 2. Profundidad de bits (cuantización), imagen en gris ==")
    print(f"{'bits':<6}{'niveles':<9}{'valores distintos':<19}{'memoria teórica':<17}{'memoria real (uint8)'}")

    lista_bits = [8, 4, 3, 2, 1]
    pixeles = gris.shape[0] * gris.shape[1]

    fig2, ejes2 = plt.subplots(3, 5, figsize=(17, 10), layout="constrained")  # fila 0: gris, 1: histograma, 2: color
    fig2.suptitle("Paso 2b · Profundidad de bits: mismos píxeles, cada vez menos niveles")

    for col, bits in enumerate(lista_bits):
        g = reducir_bits(gris, bits)
        c = reducir_bits(imagen, bits)
        distintos = len(np.unique(g))
        teorica_kb = pixeles * bits / 8 / 1024
        real_kb = g.nbytes / 1024
        print(f"{bits:<6}{2 ** bits:<9}{distintos:<19}{f'{teorica_kb:.0f} KB':<17}{real_kb:.0f} KB")

        # fila 0: imagen en gris
        ejes2[0, col].imshow(g, cmap="gray", vmin=0, vmax=255)
        ejes2[0, col].set_title(f"Gris {bits} bit(s) = {2 ** bits} niveles")
        ejes2[0, col].axis("off")

        # fila 1: histograma del gris
        ejes2[1, col].hist(g.ravel(), bins=256, range=(0, 256), color="gray")
        ejes2[1, col].set_title(f"Histograma: {distintos} valores distintos")
        ejes2[1, col].set_yticks([])

        # fila 2: imagen a color
        ejes2[2, col].imshow(a_rgb(c))
        ejes2[2, col].set_title(f"Color {bits}x3 = {bits * 3} bits = {2 ** (bits * 3):,} colores")
        ejes2[2, col].axis("off")

    # d) Un solo plt.show() para ver las dos figuras
    plt.show()