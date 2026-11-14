"""Práctica 1 · Paso 3 · Tipos de imagen y canales de color
==========================================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Comparar los tres tipos de imagen (color, gris y binaria), separar los
canales B, G y R, calcular el gris de varias formas y usar una máscara
binaria para recortar la imagen.

Este script parte del Paso 1: conserva la sección [BASE] y cambia el
programa principal.

Ejecutar
--------
    python p1_paso3_tipos_canales.py
    python p1_paso3_tipos_canales.py --imagen otra_foto.png
    python p1_paso3_tipos_canales.py --guardar
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =====================================================================================
# [BASE] Configuración y funciones que se reutilizan en los pasos 2 y 3
# =====================================================================================
CARPETA = Path(__file__).resolve().parent
RUTA_IMAGEN = CARPETA / "recursos" / "mi_foto.png"
CARPETA_CAPTURAS = CARPETA / "reporte"
MODO_GUARDAR = "--guardar" in sys.argv


def cargar_imagen(ruta):
    """Lee una imagen del disco. OpenCV la devuelve en orden de canales B, G, R y tipo uint8."""
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        sys.exit(f"No se pudo leer la imagen: {ruta}")
    return imagen


def capturar_webcam(indice=0):
    """Abre la cámara y devuelve una foto. ESPACIO = capturar · ESC = cancelar."""
    camara = cv2.VideoCapture(indice)
    if not camara.isOpened():
        sys.exit("No se pudo abrir la cámara (¿la usa otra aplicación?)")
    foto = None
    while True:
        ok, frame = camara.read()
        if not ok:
            break
        cv2.imshow("Webcam - ESPACIO captura, ESC cancela", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 32:
            foto = frame.copy()
            break
        if tecla == 27:
            break
    camara.release()
    cv2.destroyAllWindows()
    if foto is None:
        sys.exit("No se capturó ninguna foto")
    return foto


def ruta_argumento(nombre):
    """Devuelve el valor que sigue a un argumento, o None si no se escribió."""
    if nombre not in sys.argv:
        return None
    i = sys.argv.index(nombre)
    if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("--"):
        return None
    return sys.argv[i + 1]


def obtener_imagen():
    """Elige la fuente: webcam con --camara, una ruta con --imagen o el archivo RUTA_IMAGEN."""
    if "--camara" in sys.argv:
        return capturar_webcam()
    ruta = ruta_argumento("--imagen")
    return cargar_imagen(Path(ruta) if ruta else RUTA_IMAGEN)


def imprimir_info(nombre, img):
    """Imprime las propiedades básicas de una imagen (sirve para color, gris o binaria)."""
    alto, ancho = img.shape[:2]
    canales = 1 if img.ndim == 2 else img.shape[2]
    print(f"{nombre:<10} shape={str(img.shape):<15} ancho={ancho:<4} alto={alto:<4} "
          f"canales={canales}  dtype={img.dtype}  min={img.min():>3}  max={img.max():>3}  "
          f"memoria={img.nbytes / 1024:.0f} KB")


def a_rgb(img):
    """Matplotlib espera R, G, B; OpenCV entrega B, G, R. Las imágenes en gris se dejan igual."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    imagen = obtener_imagen()
    print("== Imagen de trabajo ==")
    imprimir_info("color", imagen)

    # ---------------------------------------------------------------------------------
    # 1. Color = 3 canales: separar B, G y R
    # ---------------------------------------------------------------------------------
    b, g, r = cv2.split(imagen)                 # 3 matrices de alto x ancho
    print("\n== 1. Canales de la imagen a color ==")
    imprimir_info("canal B", b)
    imprimir_info("canal G", g)
    imprimir_info("canal R", r)

    ceros = np.zeros_like(b)
    solo_b = cv2.merge([b, ceros, ceros])       # solo el canal azul, en su color
    solo_g = cv2.merge([ceros, g, ceros])       # solo el canal verde
    solo_r = cv2.merge([ceros, ceros, r])       # solo el canal rojo, en su color
    reconstruida = cv2.merge([b, g, r])         # comprueba que separar y unir no cambia nada
    print(f"merge([b, g, r]) es idéntica a la original: {np.array_equal(reconstruida, imagen)}")

    fig_canales, ejes = plt.subplots(2, 4, figsize=(18, 8.5), layout="constrained")
    fig_canales.suptitle("Paso 3a · La imagen a color y sus tres canales",
                         fontsize=14, fontweight="bold")
    ejes[0, 0].imshow(a_rgb(imagen)); ejes[0, 0].set_title("Original (RGB)")
    ejes[0, 1].imshow(b, cmap="gray", vmin=0, vmax=255); ejes[0, 1].set_title("Canal B (gris)")
    ejes[0, 2].imshow(g, cmap="gray", vmin=0, vmax=255); ejes[0, 2].set_title("Canal G (gris)")
    ejes[0, 3].imshow(r, cmap="gray", vmin=0, vmax=255); ejes[0, 3].set_title("Canal R (gris)")

    ejes[1, 0].imshow(imagen)                   # a propósito: BGR mostrado como RGB
    ejes[1, 0].set_title("BGR sin convertir (error)")
    ejes[1, 1].imshow(a_rgb(solo_b)); ejes[1, 1].set_title("Canal B en su color")
    ejes[1, 2].imshow(a_rgb(solo_g)); ejes[1, 2].set_title("Canal G en su color")
    ejes[1, 3].imshow(a_rgb(solo_r)); ejes[1, 3].set_title("Canal R en su color")
    for eje in ejes.flat:
        eje.axis("off")

    # ---------------------------------------------------------------------------------
    # 2. Gris = 1 canal: cuatro formas de calcularlo
    # ---------------------------------------------------------------------------------
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris_formula = np.round(0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
    promedio = ((b.astype(np.uint16) + g + r) // 3).astype(np.uint8)
    with np.errstate(over="warn"):
        promedio_mal = (b + g + r) // 3          # ¡a propósito! uint8 se desborda

    print("\n== 2. Escala de grises ==")
    imprimir_info("gris", gris)
    print(f"diferencia máxima cvtColor vs fórmula: {cv2.absdiff(gris, gris_formula).max()}")

    y, x = 60, 250
    suma_real = int(b[y, x]) + int(g[y, x]) + int(r[y, x])
    suma_uint8 = int(np.uint8(b[y, x] + g[y, x] + r[y, x]))
    print(f"píxel (y={y}, x={x}): B={b[y, x]} G={g[y, x]} R={r[y, x]}")
    print(f"suma real = {suma_real}   suma en uint8 = {suma_uint8}  <- desbordamiento (módulo 256)")
    for nombre, img in (("cvtColor", gris), ("fórmula", gris_formula),
                        ("promedio", promedio), ("promedio_mal", promedio_mal)):
        print(f"{nombre:<13} min={img.min():>3}  max={img.max():>3}  "
              f"media={img.mean():6.1f}  desv={img.std():5.1f}")

    fig_gris, ejes = plt.subplots(1, 4, figsize=(18, 5), layout="constrained")
    fig_gris.suptitle("Paso 3b · Cuatro formas de calcular el gris (la última se desborda)",
                      fontsize=14, fontweight="bold")
    for eje, (titulo, img) in zip(ejes, (
            ("cv2.cvtColor (0.299R+0.587G+0.114B)", gris),
            ("Fórmula explícita", gris_formula),
            ("Promedio con uint16 (correcto)", promedio),
            ("Promedio con uint8 (desbordado)", promedio_mal))):
        eje.imshow(img, cmap="gray", vmin=0, vmax=255)
        eje.set_title(titulo, fontsize=10)
        eje.axis("off")

    # ---------------------------------------------------------------------------------
    # 3. Binaria = 2 valores: umbral y máscara
    # ---------------------------------------------------------------------------------
    _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
    print("\n== 3. Imagen binaria ==")
    imprimir_info("binaria", binaria)
    print(f"valores distintos: {np.unique(binaria)}")
    blancos = cv2.countNonZero(binaria)
    print(f"píxeles blancos: {blancos} de {binaria.size} ({100 * blancos / binaria.size:.1f} %)")

    mascara_bool = binaria.astype(bool)
    coincide = np.array_equal(mascara_bool, gris > 127)
    print(f"máscara NumPy: {mascara_bool.dtype} {mascara_bool.shape} -> coincide: {coincide}")
    recorte = cv2.bitwise_and(imagen, imagen, mask=binaria)

    fig_bin, ejes = plt.subplots(1, 5, figsize=(20, 4.8), layout="constrained")
    fig_bin.suptitle("Paso 3c · La binaria depende del umbral y sirve como máscara",
                     fontsize=14, fontweight="bold")
    ejes[0].imshow(gris, cmap="gray", vmin=0, vmax=255); ejes[0].set_title("Gris")
    for eje, t in zip(ejes[1:4], (64, 127, 192)):
        _, m = cv2.threshold(gris, t, 255, cv2.THRESH_BINARY)
        eje.imshow(m, cmap="gray", vmin=0, vmax=255)
        eje.set_title(f"Binaria T={t}\n{100 * cv2.countNonZero(m) / m.size:.1f} % blanco")
    ejes[4].imshow(a_rgb(recorte)); ejes[4].set_title("Recorte con la máscara T=127")
    for eje in ejes:
        eje.axis("off")

    # ---------------------------------------------------------------------------------
    # 4. Resumen de los tres tipos de imagen
    # ---------------------------------------------------------------------------------
    distintos_color = len(np.unique(imagen.reshape(-1, 3), axis=0))
    print("\n== 4. Resumen de tipos de imagen ==")
    print(f"{'tipo':<8} {'shape':<16} {'dtype':<7} {'valores distintos':>18} {'memoria':>9}")
    for nombre, img, distintos in (("color", imagen, distintos_color),
                                   ("gris", gris, len(np.unique(gris))),
                                   ("binaria", binaria, len(np.unique(binaria)))):
        print(f"{nombre:<8} {str(img.shape):<16} {str(img.dtype):<7} "
              f"{distintos:>18} {img.nbytes / 1024:>7.0f} KB")

    fig_resumen, ejes = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")
    fig_resumen.suptitle("Paso 3d · Resumen: color, gris y binaria",
                         fontsize=14, fontweight="bold")
    ejes[0].imshow(a_rgb(imagen)); ejes[0].set_title(f"Color · {imagen.shape} · {distintos_color} colores")
    ejes[1].imshow(gris, cmap="gray", vmin=0, vmax=255); ejes[1].set_title("Gris · 1 canal · 256 niveles")
    ejes[2].imshow(binaria, cmap="gray", vmin=0, vmax=255); ejes[2].set_title("Binaria · 2 valores · máscara")
    for eje in ejes:
        eje.axis("off")

    if MODO_GUARDAR:
        CARPETA_CAPTURAS.mkdir(exist_ok=True)
        fig_canales.savefig(CARPETA_CAPTURAS / "p1_paso3_canales.png", dpi=150, bbox_inches="tight")
        fig_gris.savefig(CARPETA_CAPTURAS / "p1_paso3_gris.png", dpi=150, bbox_inches="tight")
        fig_bin.savefig(CARPETA_CAPTURAS / "p1_paso3_binaria.png", dpi=150, bbox_inches="tight")
        fig_resumen.savefig(CARPETA_CAPTURAS / "p1_paso3_resumen.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        print("\n[capturas] figuras guardadas en reporte/p1_paso3_*.png")
    else:
        plt.show()
