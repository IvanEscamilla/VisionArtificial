"""Práctica 1 · Paso 2 · Resolución espacial y profundidad de bits
=================================================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Ver los dos pasos de la digitalización: el muestreo (resolución espacial)
y la cuantización (profundidad de bits), y cómo cada uno pierde información.

Este script parte del Paso 1: conserva la sección [BASE] y cambia el
programa principal.

Ejecutar
--------
    python p1_paso2_resolucion_bits.py
    python p1_paso2_resolucion_bits.py --imagen otra_foto.png
    python p1_paso2_resolucion_bits.py --guardar
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
# Funciones nuevas del paso 2
# =====================================================================================
def reducir_resolucion(img, factor):
    """Muestreo: deja 1 píxel por cada bloque de factor x factor.

    Devuelve la imagen pequeña y la misma ampliada al tamaño original para
    compararlas lado a lado.
    """
    alto, ancho = img.shape[:2]
    pequena = cv2.resize(img, (ancho // factor, alto // factor),   # OJO: dsize va como (ANCHO, ALTO)
                         interpolation=cv2.INTER_AREA)              # INTER_AREA promedia el bloque (reducir)
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
    imagen = obtener_imagen()
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    print("== Imagen de trabajo ==")
    imprimir_info("color", imagen)
    imprimir_info("gris", gris)

    # ---------------------------------------------------------------------------------
    # 1. Resolución espacial: se mantienen los 8 bits y cambia el número de píxeles
    # ---------------------------------------------------------------------------------
    factores = [1, 2, 4, 8, 16, 32]
    resultados_resolucion = []
    print("\n== 1. Resolución espacial (muestreo), imagen a color de 8 bits por canal ==")
    print(f"{'factor':>6}  {'resolución':>10}  {'píxeles':>9}  {'memoria':>10}")
    for factor in factores:
        pequena, ampliada = reducir_resolucion(imagen, factor)
        pixeles = pequena.shape[0] * pequena.shape[1]
        memoria = pequena.nbytes / 1024
        resultados_resolucion.append((factor, pequena, ampliada))
        print(f"{factor:>6}  {pequena.shape[1]:>4}x{pequena.shape[0]:<4}  {pixeles:>9}  {memoria:>8.1f} KB")

    fig_res, ejes = plt.subplots(2, 3, figsize=(13, 9), layout="constrained")
    fig_res.suptitle("Paso 2 · Resolución espacial: de 600×400 a 18×12 píxeles",
                     fontsize=14, fontweight="bold")
    for eje, (factor, pequena, ampliada) in zip(ejes.flat, resultados_resolucion):
        eje.imshow(a_rgb(ampliada))
        eje.set_title(f"factor {factor} · {pequena.shape[1]}×{pequena.shape[0]} px")
        eje.axis("off")

    # ---------------------------------------------------------------------------------
    # 2. Profundidad de bits: se mantienen los píxeles y cambia el número de niveles
    # ---------------------------------------------------------------------------------
    valores_bits = [8, 4, 3, 2, 1]
    resultados_bits = []
    print("\n== 2. Profundidad de bits (cuantización), imagen en gris ==")
    print(f"{'bits':>4}  {'niveles':>7}  {'valores distintos':>17}  "
          f"{'memoria teórica':>16}  {'memoria real (uint8)':>21}")
    for bits in valores_bits:
        g = reducir_bits(gris, bits)
        c = reducir_bits(imagen, bits)
        niveles = 2 ** bits
        distintos = len(np.unique(g))
        memoria_teorica = gris.size * bits / 8 / 1024
        print(f"{bits:>4}  {niveles:>7}  {distintos:>17}  "
              f"{memoria_teorica:>13.0f} KB  {g.nbytes / 1024:>18.0f} KB")
        resultados_bits.append((bits, g, c))

    fig_bits, ejes = plt.subplots(3, 5, figsize=(18, 9), layout="constrained")
    fig_bits.suptitle("Paso 2 · Profundidad de bits: gris, histograma y color",
                      fontsize=14, fontweight="bold")
    for col, (bits, g, c) in enumerate(resultados_bits):
        ejes[0, col].imshow(g, cmap="gray", vmin=0, vmax=255)
        ejes[0, col].set_title(f"{bits} bits · gris")
        ejes[0, col].axis("off")

        ejes[1, col].hist(g.ravel(), bins=256, range=(0, 256), color="#333333")
        ejes[1, col].set_title(f"Histograma · {len(np.unique(g))} niveles")
        ejes[1, col].set_xlim(0, 255)
        ejes[1, col].set_yticks([])

        ejes[2, col].imshow(a_rgb(c))
        ejes[2, col].set_title(f"{bits} bits · color")
        ejes[2, col].axis("off")

    if MODO_GUARDAR:
        CARPETA_CAPTURAS.mkdir(exist_ok=True)
        fig_res.savefig(CARPETA_CAPTURAS / "p1_paso2_resolucion.png", dpi=150, bbox_inches="tight")
        fig_bits.savefig(CARPETA_CAPTURAS / "p1_paso2_bits.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        print("\n[capturas] figuras guardadas en reporte/p1_paso2_resolucion.png y p1_paso2_bits.png")
    else:
        plt.show()
