import argparse
from pathlib import Path

import cv2
import numpy as np

from p3_paso1_umbral import limpiar


VENTANA = "P3 - Paso 4: Conocimiento del dominio"
VENTANA_MASCARAS = "P3 - Paso 4: Mascaras"
ESTADO = {"hsv": None, "listo": False}


def mascara_color(frame, h_min, h_max, s_min, v_min):
    """Obtiene una máscara HSV dentro del rango de tono, saturación y brillo."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_min = int(np.clip(h_min, 0, 179))
    h_max = int(np.clip(h_max, 0, 179))
    s_min = int(np.clip(s_min, 0, 255))
    v_min = int(np.clip(v_min, 0, 255))

    # El tono HSV de OpenCV es circular: un intervalo que cruza 179/0
    # se representa como la unión de los extremos del rango.
    if h_min <= h_max:
        return cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, 255, 255))
    rango1 = cv2.inRange(hsv, (h_min, s_min, v_min), (179, 255, 255))
    rango2 = cv2.inRange(hsv, (0, s_min, v_min), (h_max, 255, 255))
    return cv2.bitwise_or(rango1, rango2)


def describir(contorno):
    """Devuelve área, circularidad y rectángulo delimitador del contorno."""
    area = cv2.contourArea(contorno)
    casco = cv2.convexHull(contorno)
    perimetro = cv2.arcLength(casco, True)
    circularidad = 4 * np.pi * area / (perimetro ** 2) if perimetro > 0 else 0
    return area, circularidad, cv2.boundingRect(contorno)


def segmentar_dominio(
    frame, rango=(5, 25, 120, 80), area_min=1500, circ_min=0.90, k=5
):
    """Segmenta por color y filtra contornos por tamaño y circularidad.

    Devuelve (máscara de color, máscara final, aceptados, rechazados).
    """
    color = limpiar(mascara_color(frame, *rango), k)
    contornos, _ = cv2.findContours(
        color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    aceptados, rechazados = [], []
    for contorno in contornos:
        area, circularidad, caja = describir(contorno)
        elemento = (contorno, area, circularidad, caja)
        if area >= area_min and circularidad >= circ_min:
            aceptados.append(elemento)
        else:
            rechazados.append(elemento)

    final = np.zeros_like(color)
    cv2.drawContours(final, [elemento[0] for elemento in aceptados], -1, 255, cv2.FILLED)
    return color, final, aceptados, rechazados


def valor_barra(nombre):
    return cv2.getTrackbarPos(nombre, VENTANA)


def ajustar_tono_por_clic(evento, x, y, banderas, parametro):
    """Al hacer clic, lee el color HSV y ajusta el rango H alrededor del tono."""
    if evento != cv2.EVENT_LBUTTONDOWN or ESTADO["hsv"] is None:
        return
    hsv = ESTADO["hsv"]
    if x < 0 or y < 0 or y >= hsv.shape[0] or x >= hsv.shape[1]:
        return
    h, s, v = (int(valor) for valor in hsv[y, x])
    h_min = max(0, h - 10)
    h_max = min(179, h + 10)
    if ESTADO["listo"]:
        cv2.setTrackbarPos("H min", VENTANA, h_min)
        cv2.setTrackbarPos("H max", VENTANA, h_max)
    print(f"Clic en ({x}, {y}): H={h} S={s} V={v}; rango H={h_min}..{h_max}")


def dibujar_resultado(frame, aceptados, rechazados):
    """Dibuja aceptados en verde y rechazados en rojo, con medidas."""
    salida = frame.copy()
    for color, elementos, estado in (
        ((0, 255, 0), aceptados, "aceptado"),
        ((0, 0, 255), rechazados, "rechazado"),
    ):
        for contorno, area, circularidad, (x, y, ancho, alto) in elementos:
            cv2.drawContours(salida, [contorno], -1, color, 2)
            cv2.rectangle(salida, (x, y), (x + ancho, y + alto), color, 2)
            etiqueta = f"{estado}: A={area:.0f} C={circularidad:.2f}"
            y_texto = max(18, y - 6)
            cv2.putText(
                salida,
                etiqueta,
                (x, y_texto),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        salida,
        f"Aceptados: {len(aceptados)}  Rechazados: {len(rechazados)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return salida


def mostrar_mascaras(mascara_color_img, mascara_final):
    """Muestra lado a lado la máscara de color y la máscara final."""
    alto, ancho = mascara_color_img.shape[:2]
    color_bgr = cv2.cvtColor(mascara_color_img, cv2.COLOR_GRAY2BGR)
    final_bgr = cv2.cvtColor(mascara_final, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        color_bgr, "Mascara de color", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        final_bgr, "Mascara final", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 255), 2, cv2.LINE_AA
    )
    return np.hstack((color_bgr, final_bgr))


def main():
    parser = argparse.ArgumentParser(
        description="Practica 3, paso 4: segmentacion usando conocimiento del dominio."
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
    cv2.createTrackbar("H min", VENTANA, 5, 179, lambda _: None)
    cv2.createTrackbar("H max", VENTANA, 25, 179, lambda _: None)
    cv2.createTrackbar("S min", VENTANA, 120, 255, lambda _: None)
    cv2.createTrackbar("V min", VENTANA, 80, 255, lambda _: None)
    cv2.createTrackbar("Area min", VENTANA, 1500, 20000, lambda _: None)
    cv2.createTrackbar("Circ x100", VENTANA, 90, 100, lambda _: None)
    cv2.setMouseCallback(VENTANA, ajustar_tono_por_clic)
    ESTADO["listo"] = True

    ultimo = None
    while True:
        ok, frame = captura.read()
        if not ok:
            break
        ESTADO["hsv"] = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        rango = (
            valor_barra("H min"),
            valor_barra("H max"),
            valor_barra("S min"),
            valor_barra("V min"),
        )
        area_min = valor_barra("Area min")
        circ_min = valor_barra("Circ x100") / 100.0

        mascara_color_img, mascara_final, aceptados, rechazados = segmentar_dominio(
            frame,
            rango=rango,
            area_min=area_min,
            circ_min=circ_min,
            k=5,
        )
        anotado = dibujar_resultado(frame, aceptados, rechazados)
        cv2.imshow(VENTANA, anotado)
        cv2.imshow(
            VENTANA_MASCARAS,
            mostrar_mascaras(mascara_color_img, mascara_final),
        )

        ultimo = {
            "aceptados": aceptados,
            "rechazados": rechazados,
            "rango": rango,
            "area_min": area_min,
            "circ_min": circ_min,
        }

        tecla = cv2.waitKey(espera_ms) & 0xFF
        if tecla == ord("q") or tecla == 27:  # q o Esc
            break

    captura.release()
    cv2.destroyAllWindows()

    if ultimo:
        print("\n== Ultimo frame procesado ==")
        print(
            f"Rango HSV: H={ultimo['rango'][0]}..{ultimo['rango'][1]}, "
            f"S>={ultimo['rango'][2]}, V>={ultimo['rango'][3]}"
        )
        print(
            f"Filtros: area minima={ultimo['area_min']}, "
            f"circularidad minima={ultimo['circ_min']:.2f}"
        )
        for estado, elementos in (
            ("aceptado", ultimo["aceptados"]),
            ("rechazado", ultimo["rechazados"]),
        ):
            for _, area, circularidad, _ in elementos:
                print(f"{estado:10s} area={area:.0f} circularidad={circularidad:.2f}")
        print(
            f"Conteo: {len(ultimo['aceptados'])} aceptados, "
            f"{len(ultimo['rechazados'])} rechazados"
        )
    else:
        print("No se recibieron frames de la fuente de video.")


if __name__ == "__main__":
    main()
