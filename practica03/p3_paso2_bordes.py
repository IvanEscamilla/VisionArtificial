import argparse
from pathlib import Path

import cv2
import numpy as np

from p3_paso1_umbral import preparar_gris


VENTANA = "P3 - Paso 2: Bordes"


def gradiente_sobel(gris):
    """Calcula derivadas horizontal y vertical, y la magnitud del gradiente."""
    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    magnitud = cv2.magnitude(gx, gy)
    return gx, gy, magnitud


def a_8bits(matriz):
    """Convierte la magnitud absoluta de una matriz a escala visible 0..255."""
    return cv2.normalize(
        np.abs(matriz), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)


def segmentar_bordes(gris, bajo=50, alto=150, k_cierre=7):
    """Canny, cierre de huecos y relleno de contornos externos.

    Devuelve las imágenes de bordes, bordes cerrados y regiones rellenas.
    """
    bajo = int(bajo)
    alto = int(alto)
    if bajo > alto:
        bajo, alto = alto, bajo

    bordes = cv2.Canny(gris, bajo, alto)

    k_cierre = max(1, int(k_cierre))
    if k_cierre % 2 == 0:
        k_cierre += 1
    elemento = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k_cierre, k_cierre)
    )
    cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, elemento)

    contornos, _ = cv2.findContours(
        cerrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    regiones = np.zeros_like(gris)
    cv2.drawContours(regiones, contornos, -1, 255, cv2.FILLED)
    return bordes, cerrados, regiones


def con_titulo(imagen, titulo):
    """Agrega una franja con título a una imagen BGR o en escala de grises."""
    if len(imagen.shape) == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        imagen = imagen.copy()
    cv2.rectangle(imagen, (0, 0), (imagen.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        imagen,
        titulo,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return imagen


def crear_mosaico(paneles):
    """Acomoda seis paneles en una cuadrícula de 2 por 3."""
    alto = min(panel.shape[0] for panel in paneles)
    ancho = min(panel.shape[1] for panel in paneles)
    ajustados = [cv2.resize(p, (ancho, alto)) for p in paneles]
    return np.vstack((np.hstack(ajustados[:3]), np.hstack(ajustados[3:])))


def valor_barra(nombre):
    return cv2.getTrackbarPos(nombre, VENTANA)


def main():
    parser = argparse.ArgumentParser(
        description="Practica 3, paso 2: segmentacion por fronteras."
    )
    parser.add_argument(
        "--video",
        nargs="?",
        const="recursos/video_sintetico.mp4",
        default=None,
        help=(
            "Ruta del video. Si se escribe --video sin ruta, usa "
            "recursos/video_sintetico.mp4. Si se omite, abre la webcam."
        ),
    )
    parser.add_argument(
        "--camara", type=int, default=0, help="Indice de webcam (por defecto: 0)."
    )
    args = parser.parse_args()

    if args.video:
        ruta = Path(args.video)
        if not ruta.exists():
            raise SystemExit(
                f"No se encontro el video: {ruta}\n"
                "Ejecuta desde practica03 y comprueba la carpeta recursos."
            )
        captura = cv2.VideoCapture(str(ruta))
    else:
        captura = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW)

    if not captura.isOpened():
        raise SystemExit(
            "No se pudo abrir la fuente de video. Revisa la ruta del archivo "
            "o el indice de la webcam."
        )

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Suavizado x10", VENTANA, 15, 50, lambda _: None)
    cv2.createTrackbar("Canny bajo", VENTANA, 50, 255, lambda _: None)
    cv2.createTrackbar("Canny alto", VENTANA, 150, 255, lambda _: None)
    cv2.createTrackbar("Cierre", VENTANA, 7, 31, lambda _: None)

    ultimo = None
    while True:
        ok, frame = captura.read()
        if not ok:
            break

        sigma = valor_barra("Suavizado x10") / 10.0
        bajo = valor_barra("Canny bajo")
        alto = valor_barra("Canny alto")
        k_cierre = max(1, valor_barra("Cierre"))

        gris = preparar_gris(frame, sigma)
        gx, gy, magnitud = gradiente_sobel(gris)
        bordes, cerrados, regiones = segmentar_bordes(
            gris, bajo, alto, k_cierre
        )

        paneles = [
            con_titulo(a_8bits(gx), "Sobel |Gx|"),
            con_titulo(a_8bits(gy), "Sobel |Gy|"),
            con_titulo(a_8bits(magnitud), "Magnitud del gradiente"),
            con_titulo(bordes, f"Canny: {cv2.countNonZero(bordes)} pixeles"),
            con_titulo(cerrados, "Bordes cerrados"),
            con_titulo(
                regiones,
                f"Regiones rellenas: {100 * cv2.countNonZero(regiones) / regiones.size:.1f}%",
            ),
        ]
        cv2.imshow(VENTANA, crear_mosaico(paneles))

        ultimo = {
            "pixeles_borde": cv2.countNonZero(bordes),
            "porcentaje_regiones": 100 * cv2.countNonZero(regiones) / regiones.size,
            "bajo": min(bajo, alto),
            "alto": max(bajo, alto),
            "k": k_cierre if k_cierre % 2 else k_cierre + 1,
        }

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q") or tecla == 27:  # q o Esc
            break

    captura.release()
    cv2.destroyAllWindows()

    if ultimo:
        print("\n== Ultimo frame procesado ==")
        print(
            f"Canny bajo = {ultimo['bajo']}  |  Canny alto = {ultimo['alto']}  "
            f"|  Cierre = {ultimo['k']}"
        )
        print(f"pixeles de borde (Canny): {ultimo['pixeles_borde']}")
        print(
            f"regiones rellenas       : {ultimo['porcentaje_regiones']:.1f} % de la imagen"
        )
    else:
        print("No se recibieron frames de la fuente de video.")


if __name__ == "__main__":
    main()
