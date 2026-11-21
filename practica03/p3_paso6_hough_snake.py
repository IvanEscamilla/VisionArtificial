import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour


VENTANA = "P3 - Paso 6: Hough y Snake"
VENTANA_SNAKE = "P3 - Snake"
ESTADO = {"gris": None, "frame": None, "lineas": [], "circulos": []}


def lineas_hough(gris, votos=80, largo_min=100):
    """Devuelve segmentos detectados con Hough: [(x1, y1, x2, y2), ...]."""
    bordes = cv2.Canny(gris, 50, 150)
    lineas = cv2.HoughLinesP(
        bordes,
        1,
        np.pi / 180,
        int(votos),
        minLineLength=int(largo_min),
        maxLineGap=10,
    )
    if lineas is None:
        return []
    return [tuple(int(v) for v in linea) for linea in lineas.reshape(-1, 4)]


def circulos_hough(gris, sensibilidad=30, r_min=15, r_max=60):
    """Devuelve círculos detectados como [(x, y, radio), ...]."""
    suave = cv2.medianBlur(gris, 5)
    circulos = cv2.HoughCircles(
        suave,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=120,
        param2=int(sensibilidad),
        minRadius=int(r_min),
        maxRadius=int(r_max),
    )
    if circulos is None:
        return []
    return [
        tuple(int(v) for v in circulo)
        for circulo in np.round(circulos.reshape(-1, 3))
    ]


def snake(gris, centro, radio, alfa=0.015, beta=10.0, gamma=0.001, puntos=200):
    """Ajusta un contorno activo iniciado como un círculo alrededor del centro.

    El centro se recibe en coordenadas OpenCV (x, y); scikit-image usa (fila, columna).
    """
    alto, ancho = gris.shape[:2]
    cx = float(np.clip(centro[0], 0, ancho - 1))
    cy = float(np.clip(centro[1], 0, alto - 1))
    radio = float(radio)
    t = np.linspace(0, 2 * np.pi, int(puntos), endpoint=False)
    inicial = np.column_stack(
        [cy + radio * np.sin(t), cx + radio * np.cos(t)]
    )
    suave = gaussian(gris, sigma=3, preserve_range=False)
    final = active_contour(
        suave,
        inicial,
        alpha=float(alfa),
        beta=float(beta),
        gamma=float(gamma),
    )
    return inicial, final


def a_puntos_cv(curva):
    """Convierte (fila, columna) de scikit-image a puntos (x, y) de OpenCV."""
    return np.round(curva[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)


def valor_barra(nombre):
    return cv2.getTrackbarPos(nombre, VENTANA)


def dibujar_hough(frame, lineas, circulos):
    """Dibuja segmentos en rojo y círculos en verde sobre una copia del frame."""
    resultado = frame.copy()
    for x1, y1, x2, y2 in lineas:
        cv2.line(resultado, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
    for x, y, radio in circulos:
        cv2.circle(resultado, (x, y), radio, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(resultado, (x, y), 2, (0, 255, 0), -1)
    cv2.putText(
        resultado,
        f"Lineas: {len(lineas)}  Circulos: {len(circulos)}  (clic para snake)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return resultado


def al_hacer_clic(evento, x, y, banderas, parametro):
    """Congela el frame actual y lanza una snake centrada en el clic."""
    if evento != cv2.EVENT_LBUTTONDOWN:
        return
    gris = ESTADO["gris"]
    frame = ESTADO["frame"]
    if gris is None or frame is None:
        return

    alto, ancho = gris.shape[:2]
    if not (0 <= x < ancho and 0 <= y < alto):
        return

    radio = max(5, valor_barra("Radio snake"))
    # Mantén el círculo inicial dentro de los límites de la imagen.
    radio = min(radio, max(3, x - 1), max(3, ancho - x - 2),
                max(3, y - 1), max(3, alto - y - 2))
    alfa = max(0.001, valor_barra("Alfa x1000") / 1000.0)
    beta = float(valor_barra("Beta"))
    centro = (x, y)
    print(f"Lanzando snake en {centro}, radio inicial={radio}, alfa={alfa:.3f}, beta={beta:.1f}")

    resultado = frame.copy()
    t0 = time.perf_counter()
    try:
        inicial, final = snake(gris, centro, radio, alfa=alfa, beta=beta)
    except Exception as error:
        print(f"No se pudo calcular la snake: {error}")
        return
    duracion_ms = (time.perf_counter() - t0) * 1000.0

    cv2.polylines(resultado, [a_puntos_cv(inicial)], True, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.polylines(resultado, [a_puntos_cv(final)], True, (0, 255, 0), 2, cv2.LINE_AA)
    radios_finales = np.sqrt(
        (final[:, 1] - x) ** 2 + (final[:, 0] - y) ** 2
    )
    radio_medio = float(np.mean(radios_finales))
    cv2.putText(
        resultado,
        f"Snake: {duracion_ms:.0f} ms, radio medio final={radio_medio:.1f}px",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(VENTANA_SNAKE, resultado)
    print(
        f"snake en ({x}, {y}): {duracion_ms:.0f} ms, "
        f"radio medio final = {radio_medio:.1f} px"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Practica 3, paso 6: transformada de Hough y contorno activo."
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
    cv2.createTrackbar("Votos", VENTANA, 80, 200, lambda _: None)
    cv2.createTrackbar("Largo min", VENTANA, 100, 400, lambda _: None)
    cv2.createTrackbar("Sensibilidad", VENTANA, 30, 100, lambda _: None)
    cv2.createTrackbar("Radio snake", VENTANA, 60, 150, lambda _: None)
    cv2.createTrackbar("Alfa x1000", VENTANA, 15, 100, lambda _: None)
    cv2.createTrackbar("Beta", VENTANA, 10, 30, lambda _: None)
    cv2.setMouseCallback(VENTANA, al_hacer_clic)

    ultimo = None
    while True:
        ok, frame = captura.read()
        if not ok:
            break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (5, 5), 1.5)

        votos = max(1, valor_barra("Votos"))
        largo_min = max(1, valor_barra("Largo min"))
        sensibilidad = max(1, valor_barra("Sensibilidad"))
        lineas = lineas_hough(gris, votos=votos, largo_min=largo_min)
        circulos = circulos_hough(gris, sensibilidad=sensibilidad)

        ESTADO["gris"] = gris.copy()
        ESTADO["frame"] = frame.copy()
        ESTADO["lineas"] = lineas
        ESTADO["circulos"] = circulos

        cv2.imshow(VENTANA, dibujar_hough(frame, lineas, circulos))
        ultimo = {"lineas": lineas, "circulos": circulos}

        tecla = cv2.waitKey(espera_ms) & 0xFF
        if tecla == ord("q") or tecla == 27:  # q o Esc
            break

    captura.release()
    cv2.destroyAllWindows()

    if ultimo:
        print("\n== Ultimo frame procesado ==")
        print(f"lineas : {ultimo['lineas']}")
        print(f"circulos: {ultimo['circulos']}")
    else:
        print("No se recibieron frames de la fuente de video.")


if __name__ == "__main__":
    main()
