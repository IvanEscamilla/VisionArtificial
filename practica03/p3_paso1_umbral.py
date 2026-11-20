import argparse
from pathlib import Path

import cv2
import numpy as np


VENTANA = "P3 - Paso 1: Umbralizacion"


def preparar_gris(frame, sigma=1.5):
    """Convierte a gris y suaviza para reducir ruido antes de umbralizar."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if sigma > 0:
        gris = cv2.GaussianBlur(gris, (0, 0), sigma)
    return gris


def segmentar_global(gris, t):
    """Umbral global: los píxeles más oscuros que t quedan blancos."""
    _, mascara = cv2.threshold(gris, t, 255, cv2.THRESH_BINARY_INV)
    return mascara


def segmentar_otsu(gris):
    """Calcula automáticamente el umbral de Otsu. Devuelve máscara y umbral."""
    t, mascara = cv2.threshold(
        gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return mascara, t


def segmentar_adaptativo(gris, bloque=101, c=10):
    """Umbral adaptativo gaussiano; bloque debe ser impar y mayor que 1."""
    bloque = max(3, int(bloque))
    if bloque % 2 == 0:
        bloque += 1
    return cv2.adaptiveThreshold(
        gris,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bloque,
        int(c),
    )


def limpiar(mascara, k=5):
    """Apertura para quitar ruido pequeño y cierre para tapar huecos."""
    k = int(k)
    if k < 2:
        return mascara
    if k % 2 == 0:
        k += 1
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    abierta = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemento)
    return cv2.morphologyEx(abierta, cv2.MORPH_CLOSE, elemento)


def porcentaje(mascara):
    """Porcentaje de píxeles blancos (objeto) en una máscara binaria."""
    return 100.0 * cv2.countNonZero(mascara) / mascara.size


def con_titulo(imagen, titulo):
    """Asegura imagen BGR y agrega una etiqueta legible."""
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
    fila1 = np.hstack(ajustados[:3])
    fila2 = np.hstack(ajustados[3:])
    return np.vstack((fila1, fila2))


def valor_barra(nombre, ventana):
    return cv2.getTrackbarPos(nombre, ventana)


def main():
    parser = argparse.ArgumentParser(
        description="Practica 3, paso 1: umbralizacion global, Otsu y adaptativa."
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
                "Comprueba que exista o ejecuta, por ejemplo:\n"
                "  python p3_paso1_umbral.py --video recursos/video_sintetico.mp4"
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
    # La barra de suavizado representa sigma en décimas: 15 equivale a 1.5.
    cv2.createTrackbar("Suavizado x10", VENTANA, 15, 50, lambda _: None)
    cv2.createTrackbar("T global", VENTANA, 150, 255, lambda _: None)
    # El valor 50 se transforma en bloque 101 (2*50 + 1).
    cv2.createTrackbar("Bloque", VENTANA, 50, 100, lambda _: None)
    cv2.createTrackbar("C", VENTANA, 10, 50, lambda _: None)
    cv2.createTrackbar("Limpieza", VENTANA, 5, 21, lambda _: None)

    ultimo = None
    while True:
        ok, frame = captura.read()
        if not ok:
            # Al terminar un archivo de video, se cierra normalmente.
            break

        sigma = valor_barra("Suavizado x10", VENTANA) / 10.0
        t_global = valor_barra("T global", VENTANA)
        bloque = 2 * max(1, valor_barra("Bloque", VENTANA)) + 1
        c = valor_barra("C", VENTANA)
        k = valor_barra("Limpieza", VENTANA)

        gris = preparar_gris(frame, sigma)
        mascara_global = segmentar_global(gris, t_global)
        mascara_otsu, t_otsu = segmentar_otsu(gris)
        mascara_adaptativa = segmentar_adaptativo(gris, bloque, c)
        adaptativa_limpia = limpiar(mascara_adaptativa, k)

        paneles = [
            con_titulo(frame, "Original"),
            con_titulo(gris, "Gris suavizado"),
            con_titulo(
                mascara_global,
                f"Global: {porcentaje(mascara_global):.1f}% (T={t_global})",
            ),
            con_titulo(
                mascara_otsu,
                f"Otsu: {porcentaje(mascara_otsu):.1f}% (T={t_otsu:.0f})",
            ),
            con_titulo(
                mascara_adaptativa,
                f"Adaptativo: {porcentaje(mascara_adaptativa):.1f}%",
            ),
            con_titulo(
                adaptativa_limpia,
                f"Adaptativo limpio: {porcentaje(adaptativa_limpia):.1f}%",
            ),
        ]
        mosaico = crear_mosaico(paneles)
        cv2.imshow(VENTANA, mosaico)

        ultimo = {
            "T global": t_global,
            "T Otsu": t_otsu,
            "Global": porcentaje(mascara_global),
            "Otsu": porcentaje(mascara_otsu),
            "Adaptativo": porcentaje(mascara_adaptativa),
            "Limpio": porcentaje(adaptativa_limpia),
        }

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q") or tecla == 27:  # q o Esc
            break

    captura.release()
    cv2.destroyAllWindows()

    if ultimo:
        print("\n== Ultimo frame procesado ==")
        print(f"T global = {ultimo['T global']}  |  T de Otsu = {ultimo['T Otsu']:.0f}")
        print(f"Global      objeto = {ultimo['Global']:.1f} % de la imagen")
        print(f"Otsu        objeto = {ultimo['Otsu']:.1f} % de la imagen")
        print(f"Adaptativo  objeto = {ultimo['Adaptativo']:.1f} % de la imagen")
        print(f"Limpio      objeto = {ultimo['Limpio']:.1f} % de la imagen")
    else:
        print("No se recibieron frames de la fuente de video.")


if __name__ == "__main__":
    main()
