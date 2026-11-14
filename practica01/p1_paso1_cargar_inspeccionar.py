"""Práctica 1 · Paso 1 · Cargar, mostrar e inspeccionar la imagen
=================================================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Cargar una imagen (de archivo o de la webcam), mostrar sus propiedades,
leer y modificar píxeles, y verla como una matriz de números.

Ejecutar
--------
    python p1_paso1_cargar_inspeccionar.py
    python p1_paso1_cargar_inspeccionar.py --camara
    python p1_paso1_cargar_inspeccionar.py --imagen otra_foto.png
    python p1_paso1_cargar_inspeccionar.py --guardar   # guarda capturas y no abre ventanas
"""
import sys                         # sys.argv (argumentos) y sys.exit (terminar con un mensaje)
from pathlib import Path           # rutas que funcionan igual en Windows, macOS y Linux

import cv2                         # OpenCV: leer, mostrar y procesar imágenes
import numpy as np                 # NumPy: una imagen de OpenCV ES un arreglo de NumPy
import matplotlib.pyplot as plt    # Matplotlib: figuras con varios paneles, títulos y números

if hasattr(sys.stdout, "reconfigure"):        # consola de Windows: conservar acentos y símbolos
    sys.stdout.reconfigure(encoding="utf-8")


# =====================================================================================
# [BASE] Configuración y funciones que se reutilizan en los pasos 2 y 3
# =====================================================================================
CARPETA = Path(__file__).resolve().parent                 # carpeta donde está este script
RUTA_IMAGEN = CARPETA / "recursos" / "mi_foto.png"           # imagen de trabajo (foto propia)
CARPETA_CAPTURAS = CARPETA / "reporte"
MODO_GUARDAR = "--guardar" in sys.argv                    # modo sin ventanas para las capturas


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
    if "--camara" in sys.argv:                            # sys.argv = lista de palabras escritas al ejecutar
        return capturar_webcam()
    ruta = ruta_argumento("--imagen")
    return cargar_imagen(Path(ruta) if ruta else RUTA_IMAGEN)


def imprimir_info(nombre, img):
    """Imprime las propiedades básicas de una imagen (sirve para color, gris o binaria)."""
    alto, ancho = img.shape[:2]                           # shape = (filas, columnas[, canales])
    canales = 1 if img.ndim == 2 else img.shape[2]        # ndim = 2 -> matriz (gris); 3 -> cubo (color)
    print(f"{nombre:<10} shape={str(img.shape):<15} ancho={ancho:<4} alto={alto:<4} "
          f"canales={canales}  dtype={img.dtype}  min={img.min():>3}  max={img.max():>3}  "
          f"memoria={img.nbytes / 1024:.0f} KB")


def a_rgb(img):
    """Matplotlib espera R, G, B; OpenCV entrega B, G, R. Las imágenes en gris se dejan igual."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def guardar_o_mostrar(fig, nombre):
    """En modo reporte guarda la figura; en modo normal la muestra en pantalla."""
    if MODO_GUARDAR:
        CARPETA_CAPTURAS.mkdir(exist_ok=True)
        fig.savefig(CARPETA_CAPTURAS / nombre, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def montaje_ventanas(imagen, gris, marcada, ruta):
    """Guarda un montaje de las tres ventanas de OpenCV (original, gris y marcada)."""
    gris_bgr = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    paneles = []
    for titulo, img in (("Original (BGR)", imagen), ("Escala de grises", gris_bgr),
                        ("Copia con cuadro rojo", marcada)):
        panel = img.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(panel, titulo, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)
        paneles.append(panel)
    cv2.imwrite(str(ruta), np.hstack(paneles))


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    # a. Obtener la imagen, convertirla a gris e imprimir sus propiedades
    imagen = obtener_imagen()
    print("== 1. Propiedades ==")
    imprimir_info("color", imagen)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imprimir_info("gris", gris)

    # b. Leer un píxel del cuerpo de la botella roja, a color y en gris
    y, x = 350, 420
    b, g, r = imagen[y, x]
    print(f"\n== 2. Píxel en (y={y}, x={x}) ==")
    print(f"color -> B={b}  G={g}  R={r}")
    print(f"gris  -> {gris[y, x]}")

    # c. Pintar un cuadro rojo de 40x40 píxeles sobre una copia (rojo en OpenCV es (0, 0, 255))
    marcada = imagen.copy()
    marcada[y:y + 40, x:x + 40] = (0, 0, 255)

    # d. Mostrar las tres imágenes con OpenCV o guardarlas para las capturas
    if MODO_GUARDAR:
        CARPETA_CAPTURAS.mkdir(exist_ok=True)
        cv2.imwrite(str(CARPETA_CAPTURAS / "p1_paso1_original.png"), imagen)
        cv2.imwrite(str(CARPETA_CAPTURAS / "p1_paso1_gris.png"), gris)
        cv2.imwrite(str(CARPETA_CAPTURAS / "p1_paso1_marcada.png"), marcada)
        montaje_ventanas(imagen, gris, marcada, CARPETA_CAPTURAS / "p1_paso1_ventanas.png")
        print("\n[capturas] ventanas guardadas en reporte/p1_paso1_ventanas.png")
    else:
        cv2.imshow("Original (color, 3 canales)", imagen)
        cv2.imshow("Escala de grises (1 canal)", gris)
        cv2.imshow("Copia con un cuadro rojo en (y=350, x=420)", marcada)
        print("\nPulsa cualquier tecla sobre una ventana de OpenCV para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # e. Recortar una región de 6x6 píxeles que empiece en (y0, x0)
    N = 6
    y0, x0 = 330, 410
    region_color = imagen[y0:y0 + N, x0:x0 + N]
    region_gris = gris[y0:y0 + N, x0:x0 + N]
    print(f"\n== 3. Región 6x6 a partir de (y={y0}, x={x0}) ==")
    print("Gris (1 número por píxel):")
    print(region_gris)
    for canal, nombre in enumerate("BGR"):
        print(f"Canal {nombre} (region_color[:, :, {canal}]):")
        print(region_color[:, :, canal])

    # f. Figura de Matplotlib con 3 paneles: imagen completa, región a color y región en gris
    fig, ejes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    fig.suptitle("Paso 1 · La imagen como matriz: región 6×6 y valores B, G, R",
                 fontsize=13, fontweight="bold")

    ejes[0].imshow(a_rgb(marcada))
    ejes[0].add_patch(plt.Rectangle((x0 - 1, y0 - 1), N, N, fill=False,
                                    edgecolor="yellow", linewidth=2))
    ejes[0].set_title(f"Imagen completa\ncuadro rojo en (y={y}, x={x}); región en amarillo")
    ejes[0].axis("off")

    ejes[1].imshow(a_rgb(region_color))
    for (fila, col), _ in np.ndenumerate(region_color[:, :, 0]):
        b_, g_, r_ = region_color[fila, col]
        texto = f"{b_}\n{g_}\n{r_}"
        ejes[1].text(col, fila, texto, ha="center", va="center",
                     fontsize=7, color="white", fontweight="bold")
    ejes[1].set_title("Región a color: B / G / R por píxel")
    ejes[1].set_xticks([]); ejes[1].set_yticks([])

    ejes[2].imshow(region_gris, cmap="gray", vmin=0, vmax=255)
    for (fila, col), valor in np.ndenumerate(region_gris):
        color = "white" if valor < 128 else "black"
        ejes[2].text(col, fila, str(valor), ha="center", va="center",
                     fontsize=8, color=color, fontweight="bold")
    ejes[2].set_title("Región en gris: un valor por píxel")
    ejes[2].set_xticks([]); ejes[2].set_yticks([])

    guardar_o_mostrar(fig, "p1_paso1_matriz.png")
    print("\n[capturas] figura de la matriz guardada en reporte/p1_paso1_matriz.png"
          if MODO_GUARDAR else "")
