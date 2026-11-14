"""Práctica 1 · Paso 3 · Tipos de imagen y canales de color
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado):
    python p1_paso3_tipos_canales.py            # usa recursos/astronauta.png
    python p1_paso3_tipos_canales.py --camara   # usa la webcam (ESPACIO captura)
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
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    print("== Imagen de trabajo ==")
    imagen = obtener_imagen()
    imprimir_info("color", imagen)

    # ---------------------------------------------------------------------------------
    # 1) Color = 3 canales: separar, reconstruir y visualizar cada canal
    # ---------------------------------------------------------------------------------
    b, g, r = cv2.split(imagen)                     # 3 matrices de alto x ancho
    print("\n== 1. Canales de la imagen a color ==")
    imprimir_info("canal B", b)
    imprimir_info("canal G", g)
    imprimir_info("canal R", r)

    ceros = np.zeros_like(b)
    solo_b = cv2.merge([b, ceros, ceros])           # solo el canal azul, en su color
    solo_g = cv2.merge([ceros, g, ceros])           # solo el canal verde, en su color
    solo_r = cv2.merge([ceros, ceros, r])           # solo el canal rojo, en su color
    reconstruida = cv2.merge([b, g, r])             # comprobación: debe ser igual a la original
    print("merge([b, g, r]) es idéntica a la original:", np.array_equal(reconstruida, imagen))

    fig1, ejes1 = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    fig1.suptitle("Paso 3a · La imagen a color y sus tres canales", fontsize=14)
    ejes1[0, 0].imshow(a_rgb(imagen));       ejes1[0, 0].set_title("Original (RGB)")
    ejes1[0, 1].imshow(b, cmap="gray", vmin=0, vmax=255); ejes1[0, 1].set_title("Canal B en gris")
    ejes1[0, 2].imshow(g, cmap="gray", vmin=0, vmax=255); ejes1[0, 2].set_title("Canal G en gris")
    ejes1[0, 3].imshow(r, cmap="gray", vmin=0, vmax=255); ejes1[0, 3].set_title("Canal R en gris")
    ejes1[1, 0].imshow(imagen);              ejes1[1, 0].set_title("BGR sin convertir (error)")
    ejes1[1, 1].imshow(a_rgb(solo_b));       ejes1[1, 1].set_title("Solo canal B, en azul")
    ejes1[1, 2].imshow(a_rgb(solo_g));       ejes1[1, 2].set_title("Solo canal G, en verde")
    ejes1[1, 3].imshow(a_rgb(solo_r));       ejes1[1, 3].set_title("Solo canal R, en rojo")
    for eje in ejes1.flat:
        eje.axis("off")

    # ---------------------------------------------------------------------------------
    # 2) Gris = 1 canal: cuatro formas de calcularlo
    # ---------------------------------------------------------------------------------
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris_formula = np.round(0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
    promedio = ((b.astype(np.uint16) + g.astype(np.uint16) + r.astype(np.uint16)) // 3).astype(np.uint8)
    promedio_mal = (b + g + r) // 3             # ¡a propósito! uint8 se desborda

    print("\n== 2. Escala de grises ==")
    imprimir_info("gris", gris)
    diferencia_max = int(cv2.absdiff(gris, gris_formula).max())
    print(f"diferencia máxima cvtColor vs fórmula: {diferencia_max}")

    yp, xp = 60, 250
    bp, gp, rp = (int(v) for v in imagen[yp, xp])
    suma_real = bp + gp + rp
    suma_uint8 = suma_real % 256                # uint8 da la vuelta al pasar de 255
    print(f"píxel (y={yp}, x={xp}): B={bp} G={gp} R={rp}")
    print(f"  suma real = {suma_real} suma en uint8 = {suma_uint8} <- desbordamiento (módulo 256)")

    fig2, ejes2 = plt.subplots(1, 4, figsize=(18, 5), layout="constrained")
    fig2.suptitle("Paso 3b · Cuatro formas de calcular el gris", fontsize=14)
    for eje, img, titulo in zip(
            ejes2,
            [gris, gris_formula, promedio, promedio_mal],
            ["cv2.cvtColor (ponderada)", "Fórmula 0.299R + 0.587G + 0.114B",
             "Promedio en uint16 (correcto)", "Promedio en uint8 (desbordado)"]):
        eje.imshow(img, cmap="gray", vmin=0, vmax=255)
        eje.set_title(titulo)
        eje.axis("off")

    # ---------------------------------------------------------------------------------
    # 3) Binaria = 2 valores: umbral y máscara
    # ---------------------------------------------------------------------------------
    _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
    print("\n== 3. Imagen binaria ==")
    imprimir_info("binaria", binaria)
    print("valores distintos:", np.unique(binaria))
    blancos = cv2.countNonZero(binaria)
    print(f"píxeles blancos: {blancos} de {binaria.size} "
          f"({100 * blancos / binaria.size:.1f} %)")

    mascara_bool = binaria > 0
    coincide = np.array_equal(mascara_bool, gris > 127)
    print(f"máscara NumPy: bool {mascara_bool.shape} -> coincide: {coincide}")
    recorte = cv2.bitwise_and(imagen, imagen, mask=binaria)

    fig3, ejes3 = plt.subplots(1, 5, figsize=(20, 4.5), layout="constrained")
    fig3.suptitle("Paso 3c · La binaria depende del umbral y sirve como máscara", fontsize=14)
    ejes3[0].imshow(gris, cmap="gray", vmin=0, vmax=255); ejes3[0].set_title("Gris")
    for eje, t in zip(ejes3[1:4], [64, 127, 192]):
        _, m = cv2.threshold(gris, t, 255, cv2.THRESH_BINARY)
        eje.imshow(m, cmap="gray", vmin=0, vmax=255)
        eje.set_title(f"T = {t} · {100 * cv2.countNonZero(m) / m.size:.1f} % blanco")
    ejes3[4].imshow(a_rgb(recorte)); ejes3[4].set_title("Recorte con la máscara T = 127")
    for eje in ejes3:
        eje.axis("off")

    # ---------------------------------------------------------------------------------
    # 4) Resumen de los tres tipos de imagen
    # ---------------------------------------------------------------------------------
    print("\n== 4. Resumen de tipos de imagen ==")
    print(f"{'tipo':<8} {'shape':<16} {'dtype':<6} {'valores distintos':>18} {'memoria':>10}")
    tipos = [
        ("color", imagen, len(np.unique(imagen.reshape(-1, 3), axis=0))),
        ("gris", gris, len(np.unique(gris))),
        ("binaria", binaria, len(np.unique(binaria))),
    ]
    for nombre, img, distintos in tipos:
        print(f"{nombre:<8} {str(img.shape):<16} {str(img.dtype):<6} {distintos:>18} "
              f"{img.nbytes / 1024:>7.0f} KB")

    fig4, ejes4 = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    fig4.suptitle("Paso 3d · Resumen de los tres tipos de imagen", fontsize=14)
    ejes4[0].imshow(a_rgb(imagen));              ejes4[0].set_title("Color · 3 canales")
    ejes4[1].imshow(gris, cmap="gray", vmin=0, vmax=255); ejes4[1].set_title("Gris · 1 canal · 256 niveles")
    ejes4[2].imshow(binaria, cmap="gray", vmin=0, vmax=255); ejes4[2].set_title("Binaria · 1 canal · 2 valores")
    for eje in ejes4:
        eje.axis("off")

    plt.show()
