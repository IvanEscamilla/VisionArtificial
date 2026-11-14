"""Práctica 1 · Paso 3: tipos de imagen y canales de color (color, gris y binaria).

Ejecutar (con el entorno virtual activado):
    python p1_paso3_tipos_canales.py            (usa recursos/astronauta.png)
    python p1_paso3_tipos_canales.py --camara   (ESPACIO = capturar, ESC = cancelar)
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
# PROGRAMA PRINCIPAL (solo del paso 3)
# =====================================================================================
if __name__ == "__main__":
    print("== Imagen de trabajo ==")
    imagen = obtener_imagen()
    imprimir_info("color", imagen)

    # -------------------------------------------------------------------------
    # 1. COLOR = 3 CANALES
    # -------------------------------------------------------------------------
    print("\n== 1. Canales de la imagen a color ==")
    b, g, r = cv2.split(imagen)                  # 3 matrices de alto x ancho
    imprimir_info("canal B", b)
    imprimir_info("canal G", g)
    imprimir_info("canal R", r)

    ceros = np.zeros_like(b)                     # matriz de ceros del mismo tamaño
    solo_b = cv2.merge([b, ceros, ceros])        # solo el canal azul, en su color
    solo_g = cv2.merge([ceros, g, ceros])        # solo el canal verde, en su color
    solo_r = cv2.merge([ceros, ceros, r])        # solo el canal rojo, en su color
    reconstruida = cv2.merge([b, g, r])          # une los canales otra vez
    print("merge([b, g, r]) es idéntica a la original:", np.array_equal(reconstruida, imagen))

    # Figura 2x4: arriba original y canales en gris; abajo error BGR y canales en su color
    fig1, ejes1 = plt.subplots(2, 4, figsize=(16, 8.5), layout="constrained")
    fig1.suptitle("Paso 3a · Imagen a color = 3 canales (B, G, R)")

    ejes1[0, 0].imshow(a_rgb(imagen))
    ejes1[0, 0].set_title("Original (convertida a RGB)")
    ejes1[0, 1].imshow(b, cmap="gray", vmin=0, vmax=255)
    ejes1[0, 1].set_title("Canal B (azul) en gris")
    ejes1[0, 2].imshow(g, cmap="gray", vmin=0, vmax=255)
    ejes1[0, 2].set_title("Canal G (verde) en gris")
    ejes1[0, 3].imshow(r, cmap="gray", vmin=0, vmax=255)
    ejes1[0, 3].set_title("Canal R (rojo) en gris")

    ejes1[1, 0].imshow(imagen)                   # SIN convertir: Matplotlib interpreta B,G,R como R,G,B
    ejes1[1, 0].set_title("Error típico: BGR sin convertir")
    ejes1[1, 1].imshow(a_rgb(solo_b))
    ejes1[1, 1].set_title("Canal B (azul) en su color")
    ejes1[1, 2].imshow(a_rgb(solo_g))
    ejes1[1, 2].set_title("Canal G (verde) en su color")
    ejes1[1, 3].imshow(a_rgb(solo_r))
    ejes1[1, 3].set_title("Canal R (rojo) en su color")
    for eje in ejes1.flat:
        eje.axis("off")

    # -------------------------------------------------------------------------
    # 2. GRIS = 1 CANAL
    # -------------------------------------------------------------------------
    print("\n== 2. Escala de grises ==")
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imprimir_info("gris", gris)
    gris_formula = np.round(0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
    promedio = ((b.astype(np.uint16) + g + r) // 3).astype(np.uint8)   # promedio bien hecho (uint16)
    promedio_mal = (b + g + r) // 3              # ¡a propósito! uint8 se desborda

    print("diferencia máxima cvtColor vs fórmula:", cv2.absdiff(gris, gris_formula).max())

    # Píxel (60, 250): suma real contra suma en uint8 (aquí aparece el RuntimeWarning esperado)
    pb, pg, pr = imagen[60, 250]
    suma_real = int(pb) + int(pg) + int(pr)
    suma_uint8 = pb + pg + pr                    # np.uint8 + np.uint8 -> se desborda (módulo 256)
    print(f"píxel (y=60, x=250): B={pb} G={pg} R={pr}")
    print(f"suma real = {suma_real} suma en uint8 = {suma_uint8} <- desbordamiento (módulo 256)")

    fig2, ejes2 = plt.subplots(1, 4, figsize=(16, 4.8), layout="constrained")
    fig2.suptitle("Paso 3b · Imagen en gris = 1 canal")
    for eje, img_g, titulo in zip(
        ejes2,
        [gris, gris_formula, promedio, promedio_mal],
        ["cv2.cvtColor(BGR2GRAY)",
         "0.114·B + 0.587·G + 0.299·R",
         "(B + G + R) / 3 sin pesos",
         "(B + G + R) // 3 en uint8: ERROR"],
    ):
        eje.imshow(img_g, cmap="gray", vmin=0, vmax=255)
        eje.set_title(titulo)
        eje.axis("off")

    # -------------------------------------------------------------------------
    # 3. BINARIA = 2 VALORES
    # -------------------------------------------------------------------------
    print("\n== 3. Imagen binaria ==")
    _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
    imprimir_info("binaria", binaria)
    print("valores distintos:", np.unique(binaria))        # debe dar [  0 255]
    blancos = cv2.countNonZero(binaria)
    print(f"píxeles blancos: {blancos} de {binaria.size} ({100 * blancos / binaria.size:.1f} %)")

    mascara_np = gris > 127                                # lo mismo con NumPy: matriz de True/False
    print(f"máscara NumPy: {mascara_np.dtype} {mascara_np.shape} -> coincide:",
          np.array_equal(mascara_np, binaria == 255))

    recorte = cv2.bitwise_and(imagen, imagen, mask=binaria)   # conserva solo donde la máscara no es 0

    fig3, ejes3 = plt.subplots(1, 5, figsize=(20, 4.5), layout="constrained")
    fig3.suptitle("Paso 3c · Imagen binaria = 2 valores (0 y 255), según el umbral")
    ejes3[0].imshow(gris, cmap="gray", vmin=0, vmax=255)
    ejes3[0].set_title("Gris de entrada")
    for eje, T in zip(ejes3[1:4], [64, 127, 192]):
        _, bin_t = cv2.threshold(gris, T, 255, cv2.THRESH_BINARY)
        eje.imshow(bin_t, cmap="gray", vmin=0, vmax=255)
        eje.set_title(f"Umbral T = {T}")
    ejes3[4].imshow(a_rgb(recorte))
    ejes3[4].set_title("Binaria (T=127) usada como máscara")
    for eje in ejes3:
        eje.axis("off")

    # -------------------------------------------------------------------------
    # 4. RESUMEN DE TIPOS DE IMAGEN
    # -------------------------------------------------------------------------
    print("\n== 4. Resumen de tipos de imagen ==")
    print(f"{'tipo':<9}{'shape':<17}{'dtype':<8}{'valores distintos':<19}{'memoria'}")
    colores_distintos = len(np.unique(imagen.reshape(-1, 3), axis=0))
    resumen = [
        ("color", imagen, colores_distintos),
        ("gris", gris, len(np.unique(gris))),
        ("binaria", binaria, len(np.unique(binaria))),
    ]
    for nombre, img, distintos in resumen:
        print(f"{nombre:<9}{str(img.shape):<17}{str(img.dtype):<8}{distintos:<19}{img.nbytes / 1024:.0f} KB")

    fig4, ejes4 = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    fig4.suptitle("Paso 3d · Tipos de imagen")
    textos = [
        f"Color · shape {imagen.shape}\n{colores_distintos:,} colores distintos",
        f"Gris · shape {gris.shape}\n{len(np.unique(gris))} niveles (0-255)",
        f"Binaria · shape {binaria.shape}\n{len(np.unique(binaria))} valores: 0 y 255",
    ]
    for eje, (nombre, img, _), texto in zip(ejes4, resumen, textos):
        eje.imshow(a_rgb(img), cmap="gray" if img.ndim == 2 else None, vmin=0 if img.ndim == 2 else None,
                   vmax=255 if img.ndim == 2 else None)
        eje.set_title(texto, fontsize=10)
        eje.axis("off")

    plt.show()                                   # muestra las cuatro figuras