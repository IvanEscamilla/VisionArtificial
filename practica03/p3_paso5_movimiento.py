import argparse
from pathlib import Path

import cv2
import numpy as np

from p3_paso1_umbral import limpiar, preparar_gris


VENTANA = "P3 - Paso 5: Movimiento"


def diferencia_frames(gris_anterior, gris, umbral=25):
    """Marca píxeles que cambiaron más que el umbral entre frames consecutivos."""
    diferencia = cv2.absdiff(gris, gris_anterior)
    _, mascara = cv2.threshold(diferencia, int(umbral), 255, cv2.THRESH_BINARY)
    return mascara


class FondoPromedio:
    """Fondo de promedio móvil: fondo = (1 - alfa)*fondo + alfa*frame."""

    def __init__(self):
        self.fondo = None

    def aplicar(self, gris, alfa=0.02, umbral=25):
        if self.fondo is None:
            self.fondo = gris.astype(np.float32)
        cv2.accumulateWeighted(gris, self.fondo, float(alfa))
        fondo_8 = cv2.convertScaleAbs(self.fondo)
        diferencia = cv2.absdiff(gris, fondo_8)
        _, mascara = cv2.threshold(
            diferencia, int(umbral), 255, cv2.THRESH_BINARY
        )
        return mascara, fondo_8


def cajas_en_movimiento(mascara, area_min=300):
    """Devuelve rectángulos alrededor de manchas con área suficiente."""
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [
        cv2.boundingRect(contorno)
        for contorno in contornos
        if cv2.contourArea(contorno) >= area_min
    ]


def porcentaje(mascara):
    return 100.0 * cv2.countNonZero(mascara) / mascara.size


def con_titulo(imagen, titulo):
    """Agrega un título a panel en gris o BGR."""
    if len(imagen.shape) == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        imagen = imagen.copy()
    cv2.rectangle(imagen, (0, 0), (imagen.shape[1], 29), (0, 0, 0), -1)
    cv2.putText(
        imagen,
        titulo,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return imagen


def crear_mosaico(paneles):
    """Acomoda seis paneles en una cuadrícula de dos filas y tres columnas."""
    alto = min(panel.shape[0] for panel in paneles)
    ancho = min(panel.shape[1] for panel in paneles)
    paneles = [cv2.resize(p, (ancho, alto)) for p in paneles]
    return np.vstack((np.hstack(paneles[:3]), np.hstack(paneles[3:])))


def valor_barra(nombre):
    return cv2.getTrackbarPos(nombre, VENTANA)


def main():
    parser = argparse.ArgumentParser(
        description="Practica 3, paso 5: segmentacion basada en movimiento."
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
        fps = captura.get(cv2.CAP_PROP_FPS)
        espera_ms = max(1, int(1000 / fps)) if fps and fps > 0 else 33
    else:
        captura = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW)
        espera_ms = 1

    if not captura.isOpened():
        raise SystemExit(
            "No se pudo abrir la fuente de video. Revisa la ruta del archivo "
            "o el indice de la webcam."
        )

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    # Alfa x1000: 20 representa alfa=0.020; 1 representa 0.001.
    cv2.createTrackbar("Alfa x1000", VENTANA, 20, 100, lambda _: None)
    cv2.createTrackbar("Umbral", VENTANA, 25, 255, lambda _: None)
    cv2.createTrackbar("Area min", VENTANA, 300, 10000, lambda _: None)

    promedio = FondoPromedio()
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=25, detectShadows=False
    )
    gris_anterior = None
    ultimo = None

    while True:
        ok, frame = captura.read()
        if not ok:
            break

        gris = preparar_gris(frame)
        if gris_anterior is None:
            gris_anterior = gris.copy()

        alfa = max(0.001, valor_barra("Alfa x1000") / 1000.0)
        umbral = valor_barra("Umbral")
        area_min = valor_barra("Area min")

        mascara_diferencia = diferencia_frames(gris_anterior, gris, umbral)
        mascara_promedio, fondo_promedio = promedio.aplicar(
            gris, alfa=alfa, umbral=umbral
        )
        mascara_mog2 = mog2.apply(gris)

        # Reduce ruido y pequeños parpadeos en las tres máscaras.
        mascara_diferencia = limpiar(mascara_diferencia, k=5)
        mascara_promedio = limpiar(mascara_promedio, k=5)
        mascara_mog2 = limpiar(mascara_mog2, k=5)

        fondo_mog2 = mog2.getBackgroundImage()
        if fondo_mog2 is None:
            fondo_mog2 = np.zeros_like(gris)

        cajas = cajas_en_movimiento(mascara_mog2, area_min)
        anotado = frame.copy()
        for x, y, ancho, alto in cajas:
            cv2.rectangle(anotado, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
        cv2.putText(
            anotado,
            f"Objetos en movimiento (MOG2): {len(cajas)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        paneles = [
            con_titulo(anotado, "Detecciones MOG2"),
            con_titulo(fondo_promedio, "Fondo promedio"),
            con_titulo(fondo_mog2, "Fondo MOG2"),
            con_titulo(
                mascara_diferencia,
                f"Diferencia: {porcentaje(mascara_diferencia):.2f}%",
            ),
            con_titulo(
                mascara_promedio,
                f"Promedio: {porcentaje(mascara_promedio):.2f}%",
            ),
            con_titulo(mascara_mog2, f"MOG2: {porcentaje(mascara_mog2):.2f}%"),
        ]
        cv2.imshow(VENTANA, crear_mosaico(paneles))

        ultimo = {
            "diferencia": porcentaje(mascara_diferencia),
            "promedio": porcentaje(mascara_promedio),
            "mog2": porcentaje(mascara_mog2),
            "objetos": len(cajas),
            "alfa": alfa,
            "umbral": umbral,
            "area_min": area_min,
        }
        gris_anterior = gris.copy()

        tecla = cv2.waitKey(espera_ms) & 0xFF
        if tecla == ord("q") or tecla == 27:  # q o Esc
            break

    captura.release()
    cv2.destroyAllWindows()

    if ultimo:
        print("\n== Ultimo frame procesado ==")
        print(
            f"Alfa = {ultimo['alfa']:.3f}  |  Umbral = {ultimo['umbral']}  "
            f"|  Area min = {ultimo['area_min']}"
        )
        print(f"diferencia {ultimo['diferencia']:.2f} % de la imagen en movimiento")
        print(f"promedio     {ultimo['promedio']:.2f} % de la imagen en movimiento")
        print(f"MOG2         {ultimo['mog2']:.2f} % de la imagen en movimiento")
        print(f"objetos en movimiento (MOG2): {ultimo['objetos']}")
    else:
        print("No se recibieron frames de la fuente de video.")


if __name__ == "__main__":
    main()
