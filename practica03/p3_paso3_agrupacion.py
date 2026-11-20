import argparse
from pathlib import Path
import time

import cv2
import numpy as np


VENTANA = "P3 - Paso 3: Agrupacion"
ESTADO = {"semilla": None, "celda": (320, 180)}


def segmentar_kmeans(frame, k=4, escala=0.25):
    """Agrupa píxeles por color BGR en una imagen reducida.

    Devuelve (etiquetas, pintada), ambos redimensionados al tamaño original.
    """
    k = max(2, int(k))
    escala = float(escala)
    chica = cv2.resize(
        frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA
    )
    datos = chica.reshape(-1, 3).astype(np.float32)
    criterio = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1.0,
    )
    _, etiquetas, centros = cv2.kmeans(
        datos, k, None, criterio, 1, cv2.KMEANS_PP_CENTERS
    )
    etiquetas = etiquetas.reshape(chica.shape[:2])
    pintada = centros.astype(np.uint8)[etiquetas]
    alto, ancho = frame.shape[:2]
    etiquetas = cv2.resize(
        etiquetas.astype(np.uint8),
        (ancho, alto),
        interpolation=cv2.INTER_NEAREST,
    )
    pintada = cv2.resize(
        pintada, (ancho, alto), interpolation=cv2.INTER_NEAREST
    )
    return etiquetas, pintada


def grupo_de_la_semilla(etiquetas, semilla):
    """Máscara de todos los píxeles que pertenecen al grupo de la semilla."""
    x, y = semilla
    x = int(np.clip(x, 0, etiquetas.shape[1] - 1))
    y = int(np.clip(y, 0, etiquetas.shape[0] - 1))
    return np.where(etiquetas == etiquetas[y, x], 255, 0).astype(np.uint8)


def crecer_region(frame, semilla, tolerancia=30):
    """Crece desde una semilla por píxeles conectados de color parecido."""
    alto, ancho = frame.shape[:2]
    x = int(np.clip(semilla[0], 0, ancho - 1))
    y = int(np.clip(semilla[1], 0, alto - 1))
    suave = cv2.GaussianBlur(frame, (5, 5), 0)
    mascara = np.zeros((alto + 2, ancho + 2), np.uint8)
    banderas = (
        4
        | cv2.FLOODFILL_MASK_ONLY
        | cv2.FLOODFILL_FIXED_RANGE
        | (255 << 8)
    )
    tol = (int(tolerancia),) * 3
    cv2.floodFill(suave, mascara, (x, y), 0, tol, tol, banderas)
    return mascara[1:-1, 1:-1]


def al_hacer_clic(evento, x, y, banderas, parametro):
    """Convierte un clic en cualquier panel a coordenadas del frame original."""
    if evento == cv2.EVENT_LBUTTONDOWN:
        ancho_c, alto_c = ESTADO["celda"]
        semilla = ((x % ancho_c) * 2, (y % alto_c) * 2)
        ESTADO["semilla"] = semilla
        print(f"Semilla seleccionada en {semilla}")


def con_titulo(imagen, titulo):
    """Agrega una etiqueta superior a una imagen BGR o escala de grises."""
    if len(imagen.shape) == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        imagen = imagen.copy()
    cv2.rectangle(imagen, (0, 0), (imagen.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        imagen,
        titulo,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return imagen


def main():
    parser = argparse.ArgumentParser(
        description="Practica 3, paso 3: k-means y crecimiento de regiones."
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
    cv2.createTrackbar("K", VENTANA, 4, 8, lambda _: None)
    cv2.createTrackbar("Tolerancia", VENTANA, 30, 160, lambda _: None)
    cv2.setMouseCallback(VENTANA, al_hacer_clic)

    ultimo = None
    while True:
        ok, frame = captura.read()
        if not ok:
            break

        k = max(2, cv2.getTrackbarPos("K", VENTANA))
        tolerancia = cv2.getTrackbarPos("Tolerancia", VENTANA)

        inicio = time.perf_counter()
        etiquetas, pintada = segmentar_kmeans(frame, k=k, escala=0.25)
        tiempo_ms = (time.perf_counter() - inicio) * 1000.0

        if ESTADO["semilla"] is None:
            mascara_grupo = np.zeros(frame.shape[:2], dtype=np.uint8)
            mascara_region = np.zeros(frame.shape[:2], dtype=np.uint8)
            texto_grupo = "Grupo de semilla: haz clic"
            texto_region = "Region crecida: haz clic"
        else:
            semilla = ESTADO["semilla"]
            mascara_grupo = grupo_de_la_semilla(etiquetas, semilla)
            mascara_region = crecer_region(frame, semilla, tolerancia)
            texto_grupo = f"Grupo semilla: {cv2.countNonZero(mascara_grupo)} px"
            texto_region = f"Region crecida: {cv2.countNonZero(mascara_region)} px"

        alto, ancho = frame.shape[:2]
        ancho_c = max(1, ancho // 2)
        alto_c = max(1, alto // 2)
        ESTADO["celda"] = (ancho_c, alto_c)

        paneles = [
            con_titulo(frame, "Original - haz clic en un objeto"),
            con_titulo(pintada, f"K-means (K={k}, {tiempo_ms:.1f} ms)"),
            con_titulo(mascara_grupo, texto_grupo),
            con_titulo(mascara_region, texto_region),
        ]
        paneles = [
            cv2.resize(p, (ancho_c, alto_c), interpolation=cv2.INTER_AREA)
            for p in paneles
        ]
        mosaico = np.vstack(
            (np.hstack(paneles[:2]), np.hstack(paneles[2:]))
        )
        cv2.imshow(VENTANA, mosaico)

        ultimo = {
            "k": k,
            "tiempo_ms": tiempo_ms,
            "grupo": cv2.countNonZero(mascara_grupo),
            "region": cv2.countNonZero(mascara_region),
            "semilla": ESTADO["semilla"],
        }

        tecla = cv2.waitKey(espera_ms) & 0xFF
        if tecla == ord("q") or tecla == 27:  # q o Esc
            break

    captura.release()
    cv2.destroyAllWindows()

    if ultimo:
        print("\n== Ultimo frame procesado ==")
        print(f"K = {ultimo['k']}  k-means tardo {ultimo['tiempo_ms']:.1f} ms")
        if ultimo["semilla"] is None:
            print("No se selecciono semilla; haz clic en un objeto durante la ejecucion.")
        else:
            print(f"Semilla seleccionada: {ultimo['semilla']}")
            print(f"grupo de la semilla : {ultimo['grupo']} pixeles")
            print(f"region crecida      : {ultimo['region']} pixeles")
    else:
        print("No se recibieron frames de la fuente de video.")


if __name__ == "__main__":
    main()
