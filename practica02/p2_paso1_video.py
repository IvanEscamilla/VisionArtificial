"""Práctica 2 · Paso 1 · Adquisición de video
==============================================
Alumno : Luis Fernando Salazar Torres
Carrera: Ingeniería en Inteligencia Artificial · 7° semestre
Escuela: Universidad Marista de Guadalajara

Objetivo
--------
Abrir una cámara o un archivo, leer sus propiedades, medir los FPS reales,
guardar una foto y grabar video.

Ejecutar
--------
    python p2_paso1_video.py                 # webcam
    python p2_paso1_video.py --video         # recursos/video_sintetico.mp4
    python p2_paso1_video.py --video otro.mp4
    python p2_paso1_video.py --video --guardar   # capturas y video de evidencia
"""
import sys                         # sys.argv (argumentos) y sys.exit (terminar con un mensaje)
import time                        # time.perf_counter(): reloj de alta precisión para medir FPS y ms
from datetime import datetime      # nombre único para cada grabación
from pathlib import Path           # rutas que funcionan igual en Windows, macOS y Linux

import cv2                         # OpenCV: cámara, video, filtros, segmentación y ventanas
import numpy as np                 # NumPy: cada frame es un arreglo de NumPy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =====================================================================================
# [BASE] Video: abrir la fuente, leer frames, medir FPS, escribir texto, mosaico y barras
# =====================================================================================
CARPETA = Path(__file__).resolve().parent                     # carpeta donde está este script
RUTA_VIDEO = CARPETA / "recursos" / "video_sintetico.mp4"      # video de respaldo (sin cámara)
SALIDAS = CARPETA / "salidas"                                 # aquí se guardan capturas y grabaciones
CARPETA_CAPTURAS = CARPETA / "reporte"
MODO_GUARDAR = "--guardar" in sys.argv


def abrir_fuente():
    """Webcam por defecto; con --video usa RUTA_VIDEO (o la ruta que siga a --video).

    Devuelve (captura, es_archivo, espera_ms).
    """
    if "--video" in sys.argv:                                 # ¿se pidió un archivo de video?
        i = sys.argv.index("--video")                         # posición de "--video" en la lista
        siguiente = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        ruta = siguiente if siguiente and not siguiente.startswith("--") else RUTA_VIDEO
        captura = cv2.VideoCapture(str(ruta))                 # abrir el archivo
        es_archivo = True
    else:
        captura = cv2.VideoCapture(0)                         # 0 = primera cámara del equipo
        es_archivo = False
    if not captura.isOpened():                                # no existe, está ocupada o el códec falla
        sys.exit("No se pudo abrir la fuente de video (¿cámara ocupada? prueba con --video)")
    fps = captura.get(cv2.CAP_PROP_FPS) or 30                 # algunos drivers reportan 0 -> usamos 30
    espera = max(1, int(1000 / fps)) if es_archivo else 1     # un archivo se reproduce a su velocidad real
    return captura, es_archivo, espera


def leer_frame(captura, es_archivo):
    """Lee el siguiente frame. Un archivo vuelve a empezar al terminar. Devuelve None si no hay imagen."""
    ok, frame = captura.read()                                # ok = True si llegó un frame
    if not ok and es_archivo:                                 # fin del archivo: regresamos al frame 0
        captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = captura.read()
    return frame if ok else None


class MedidorFPS:
    """Mide los cuadros por segundo reales (promedio suavizado)."""

    def __init__(self):
        self.anterior = time.perf_counter()                   # instante del frame anterior
        self.fps = 0.0

    def actualizar(self):
        ahora = time.perf_counter()
        dt = ahora - self.anterior                            # segundos entre este frame y el anterior
        self.anterior = ahora
        if dt > 0:
            self.fps = 1 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt   # suaviza los saltos
        return self.fps


def escribir(img, texto, fila=0, color=(255, 255, 255)):
    """Escribe texto en la esquina superior izquierda, sobre un rectángulo negro para que se lea siempre."""
    y = 22 + 24 * fila                                        # cada fila de texto baja 24 px
    (w, h), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)   # tamaño que ocupará el texto
    cv2.rectangle(img, (4, y - h - 5), (12 + w, y + base + 2), (0, 0, 0), -1)   # -1 = rectángulo relleno
    cv2.putText(img, texto, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def mosaico(imagenes, titulos, columnas=3, ancho=400):
    """Une varias imágenes (color o gris) en una sola, en una rejilla con título en cada celda."""
    celdas = []
    for img, titulo in zip(imagenes, titulos):
        if img.ndim == 2:                                     # gris o máscara -> 3 canales para poder unirla
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alto = int(img.shape[0] * ancho / img.shape[1])       # conservar la proporción
        celda = cv2.resize(img, (ancho, alto))                # todas las celdas del mismo tamaño
        escribir(celda, titulo, color=(0, 255, 255))          # título en amarillo
        celdas.append(celda)
    while len(celdas) % columnas:                             # completar la última fila con celdas negras
        celdas.append(np.zeros_like(celdas[-1]))
    filas = [np.hstack(celdas[i:i + columnas]) for i in range(0, len(celdas), columnas)]   # pegar en horizontal
    return np.vstack(filas)                                   # y luego las filas en vertical


def barra(nombre, ventana, inicial, maximo):
    """Crea una barra deslizante (trackbar) en una ventana que ya existe."""
    cv2.createTrackbar(nombre, ventana, inicial, maximo, lambda v: None)   # lambda: no hacer nada al moverla


def valor(nombre, ventana):
    """Lee el valor actual de una barra deslizante."""
    return cv2.getTrackbarPos(nombre, ventana)


def texto_fourcc(codigo):
    """Convierte el entero de CAP_PROP_FOURCC en las cuatro letras del códec."""
    return "".join(chr((int(codigo) >> (8 * i)) & 0xFF) for i in range(4))


def propiedades(captura, es_archivo):
    """Imprime las propiedades que reporta la fuente de video."""
    print("== 1. Propiedades de la fuente ==")
    print("tipo       :", "archivo" if es_archivo else "cámara")
    try:
        print("backend    :", captura.getBackendName())
    except Exception:
        print("backend    : (no disponible)")
    print("resolución :", int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
          int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("FPS        :", captura.get(cv2.CAP_PROP_FPS))
    print("códec      :", texto_fourcc(captura.get(cv2.CAP_PROP_FOURCC)))
    print("frames     :", int(captura.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("S = guardar frame · R = grabar / detener · Esc o Q = salir")


# =====================================================================================
# Programa principal
# =====================================================================================
def modo_interactivo():
    """Ciclo de video con teclado: S guarda una foto, R graba o detiene, Esc/Q sale."""
    captura, es_archivo, espera = abrir_fuente()
    propiedades(captura, es_archivo)

    if not es_archivo:
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("resolución pedida : 1280 x 720")
        print("resolución real  :", int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
              int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    SALIDAS.mkdir(exist_ok=True)
    medidor = MedidorFPS()
    escritor = None
    ruta_grabacion = None
    leidos = 0
    t0 = time.perf_counter()
    fps_promedio = []

    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        leidos += 1
        fps = medidor.actualizar()
        if leidos > 10:
            fps_promedio.append(fps)

        vista = frame.copy()
        escribir(vista, f"{frame.shape[1]}x{frame.shape[0]}  |  {fps:5.1f} FPS medidos")

        if escritor is not None:
            escritor.write(frame)                              # se graba el frame limpio, sin texto
            segundos = leidos / max(1.0, captura.get(cv2.CAP_PROP_FPS))
            cv2.circle(vista, (20, 50), 8, (0, 0, 255), -1)    # punto rojo de grabación
            escribir(vista, f"REC  {segundos:4.1f} s", fila=2, color=(0, 0, 255))

        cv2.imshow("Paso 1 - Video", vista)
        tecla = cv2.waitKey(espera) & 0xFF

        if tecla in (27, ord("q")):
            break
        if tecla == ord("s"):
            nombre = SALIDAS / f"frame_{leidos:05d}.png"
            cv2.imwrite(str(nombre), frame)
            print("guardado:", nombre.name)
        if tecla == ord("r"):
            if escritor is None:
                alto, ancho = frame.shape[:2]
                codigo = cv2.VideoWriter_fourcc(*"mp4v")
                ruta_grabacion = SALIDAS / f"grabacion_{datetime.now():%H%M%S}.mp4"
                escritor = cv2.VideoWriter(str(ruta_grabacion), codigo, 30, (ancho, alto))
                print("grabando en:", ruta_grabacion.name)
            else:
                escritor.release()
                escritor = None
                print(f"grabación terminada: {ruta_grabacion.name}")

    if escritor is not None:
        escritor.release()
    captura.release()
    cv2.destroyAllWindows()

    tiempo = time.perf_counter() - t0
    promedio = sum(fps_promedio) / len(fps_promedio) if fps_promedio else 0
    print("\n== 4. Resumen ==")
    print(f"frames leídos: {leidos}   tiempo: {tiempo:.1f} s   FPS promedio: {promedio:.1f}")


def modo_guardar():
    """Procesa 90 frames, guarda capturas y un video corto de evidencia, sin abrir ventanas."""
    captura, es_archivo, espera = abrir_fuente()
    propiedades(captura, es_archivo)

    if not es_archivo:
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("resolución pedida : 1280 x 720")
        print("resolución real  :", int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
              int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    SALIDAS.mkdir(exist_ok=True)
    CARPETA_CAPTURAS.mkdir(exist_ok=True)
    medidor = MedidorFPS()
    leidos = 0
    t0 = time.perf_counter()
    fps_promedio = []
    frames_guardados = []
    ruta_video = SALIDAS / "grabacion_reporte.mp4"
    escritor = None

    while leidos < 90:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        leidos += 1
        fps = medidor.actualizar()
        if leidos > 10:
            fps_promedio.append(fps)
        if es_archivo:
            time.sleep(espera / 1000)                          # reproducir a la velocidad del archivo

        if leidos == 30:
            cv2.imwrite(str(CARPETA_CAPTURAS / "p2_paso1_frame.png"), frame)
            vista = frame.copy()
            escribir(vista, f"{frame.shape[1]}x{frame.shape[0]}  |  {fps:5.1f} FPS medidos")
            cv2.imwrite(str(CARPETA_CAPTURAS / "p2_paso1_overlay.png"), vista)

            grabando = frame.copy()
            cv2.circle(grabando, (20, 50), 8, (0, 0, 255), -1)
            escribir(grabando, f"REC  1.0 s", fila=2, color=(0, 0, 255))
            cv2.imwrite(str(CARPETA_CAPTURAS / "p2_paso1_grabando.png"), grabando)

        if 25 <= leidos <= 84:
            if escritor is None:
                alto, ancho = frame.shape[:2]
                codigo = cv2.VideoWriter_fourcc(*"mp4v")
                escritor = cv2.VideoWriter(str(ruta_video), codigo, 30, (ancho, alto))
            escritor.write(frame)

        if leidos in (15, 45, 75):
            frames_guardados.append((leidos, frame.copy(), fps))

    if escritor is not None:
        escritor.release()
    captura.release()

    if frames_guardados:
        paneles = [frame.copy() for _, frame, _ in frames_guardados]
        titulos = [f"frame {n} · {fps:4.1f} FPS" for n, _, fps in frames_guardados]
        cv2.imwrite(str(CARPETA_CAPTURAS / "p2_paso1_montaje.png"),
                    mosaico(paneles, titulos, columnas=3, ancho=420))

    tiempo = time.perf_counter() - t0
    promedio = sum(fps_promedio) / len(fps_promedio) if fps_promedio else 0
    print("\n== 4. Resumen ==")
    print(f"frames leídos: {leidos}   tiempo: {tiempo:.1f} s   FPS promedio: {promedio:.1f}")
    print(f"[capturas] video de evidencia: {ruta_video}")

    if ruta_video.exists():
        verificacion = cv2.VideoCapture(str(ruta_video))
        n = int(verificacion.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_v = verificacion.get(cv2.CAP_PROP_FPS) or 30
        print(f"[capturas] {ruta_video.name}: {n} frames, {n / fps_v:.1f} s a {fps_v:.0f} FPS")
        verificacion.release()


if __name__ == "__main__":
    if MODO_GUARDAR:
        modo_guardar()
    else:
        modo_interactivo()
