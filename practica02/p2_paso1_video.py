"""Práctica 2 · Paso 1 · Adquisición de video
Visión Artificial · Lic. en Inteligencia Artificial · 7° semestre
Universidad Marista de Guadalajara
Alumno: Carlos Fernando Serrano Amador

Ejecutar (con el entorno virtual activado):
    python p2_paso1_video.py                  # webcam
    python p2_paso1_video.py --video          # recursos/video_sintetico.mp4
    python p2_paso1_video.py --video otro.mp4 # un archivo cualquiera

Teclas: S = guardar frame · R = grabar / detener · Esc o Q = salir
"""
import sys                        # sys.argv (argumentos) y sys.exit (terminar con un mensaje)
import time                       # time.perf_counter(): reloj de alta precisión para medir FPS y ms
from pathlib import Path          # rutas que funcionan igual en Windows, macOS y Linux

import cv2                        # OpenCV: cámara, video, filtros, segmentación y ventanas
import numpy as np                # NumPy: cada frame es un arreglo de NumPy

# =====================================================================================
# [BASE] Video: abrir la fuente, leer frames, medir FPS, escribir texto, mosaico y barras
# =====================================================================================
CARPETA = Path(__file__).resolve().parent                     # carpeta donde está este script
RUTA_VIDEO = CARPETA / "recursos" / "video_sintetico.mp4"     # video de respaldo (sin cámara)
SALIDAS = CARPETA / "salidas"                                 # aquí se guardan capturas y grabaciones


def abrir_fuente():
    """Webcam por defecto; con --video usa RUTA_VIDEO (o la ruta que siga a --video).
    Devuelve (captura, es_archivo, espera_ms)."""
    if "--video" in sys.argv:                                 # ¿se pidió un archivo de video?
        i = sys.argv.index("--video")                         # posición de "--video" en la lista
        ruta = sys.argv[i + 1] if i + 1 < len(sys.argv) else RUTA_VIDEO   # ruta escrita o la de respaldo
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
        self.anterior = time.perf_counter()                    # instante del frame anterior
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
    cv2.rectangle(img, (4, y - h - 5), (12 + w, y + base + 2), (0, 0, 0), -1) # -1 = rectángulo relleno
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


# =====================================================================================
# Programa principal
# =====================================================================================
if __name__ == "__main__":
    SALIDAS.mkdir(exist_ok=True)                              # la carpeta salidas/ puede no existir

    # a) Abrir la fuente e imprimir sus propiedades
    captura, es_archivo, espera = abrir_fuente()
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(captura.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    print("== 1. Propiedades de la fuente ==")
    print("tipo       :", "archivo" if es_archivo else "cámara")
    print("backend    :", captura.getBackendName())
    print("resolución :", ancho, "x", alto)
    print("FPS        :", captura.get(cv2.CAP_PROP_FPS))
    print("códec      :", codec)
    print("frames     :", int(captura.get(cv2.CAP_PROP_FRAME_COUNT)))

    # b) Si es cámara, pedir 1280x720 y comprobar qué quedó realmente
    if not es_archivo:
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("\n== 2. Resolución pedida a la cámara ==")
        print("pedida     : 1280 x 720")
        print("real       :", int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
              int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # c) Ciclo de video: leer, medir FPS, dibujar texto y mostrar
    print("\nS = guardar frame · R = grabar / detener · Esc o Q = salir")
    medidor = MedidorFPS()
    escritor = None                                           # VideoWriter mientras se graba
    ruta_grabacion = None
    inicio_grabacion = 0.0
    grabados = 0
    frames_leidos = 0
    t_inicio = time.perf_counter()

    while True:
        frame = leer_frame(captura, es_archivo)
        if frame is None:
            break
        frames_leidos += 1
        fps = medidor.actualizar()

        vista = frame.copy()
        escribir(vista, f"{frame.shape[1]}x{frame.shape[0]} | {fps:5.1f} FPS medidos")

        # Mientras se graba: escribir el frame limpio y marcar la vista con un punto rojo
        if escritor is not None:
            escritor.write(frame)                             # se guarda frame, no vista (sin texto)
            grabados += 1
            segundos = time.perf_counter() - inicio_grabacion
            cv2.circle(vista, (20, 72), 8, (0, 0, 255), -1)
            escribir(vista, f"REC {segundos:5.1f} s", 1, (0, 0, 255))

        cv2.imshow("Paso 1 - Video", vista)
        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (27, ord("q")):
            break

        # d) Teclas S y R
        if tecla == ord("s"):
            ruta = SALIDAS / f"frame_{frames_leidos:05d}.png"
            cv2.imwrite(str(ruta), frame)
            print("guardado:", ruta.name)

        elif tecla == ord("r"):
            if escritor is None:                              # empezar a grabar
                alto_f, ancho_f = frame.shape[:2]
                nombre = f"grabacion_{time.strftime('%H%M%S')}.mp4"
                ruta_grabacion = SALIDAS / nombre
                codigo = cv2.VideoWriter_fourcc(*"mp4v")
                escritor = cv2.VideoWriter(str(ruta_grabacion), codigo, 30, (ancho_f, alto_f))
                inicio_grabacion = time.perf_counter()
                grabados = 0
                print("grabando en:", nombre)
            else:                                             # detener la grabación
                escritor.release()
                escritor = None
                segundos = time.perf_counter() - inicio_grabacion
                print(f"grabación terminada: {grabados} frames ({segundos:.1f} s a 30 FPS)")

    # e) Liberar recursos y resumir
    if escritor is not None:
        escritor.release()
        print(f"grabación terminada: {grabados} frames "
              f"({time.perf_counter() - inicio_grabacion:.1f} s a 30 FPS)")
    captura.release()
    cv2.destroyAllWindows()

    duracion = time.perf_counter() - t_inicio
    print("\n== 4. Resumen ==")
    print(f"frames leídos: {frames_leidos}     tiempo: {duracion:.1f} s     "
          f"FPS promedio: {frames_leidos / max(duracion, 1e-9):.1f}")
