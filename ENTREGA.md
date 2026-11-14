# Entrega · Visión Artificial

**Alumno:** Carlos Fernando Serrano Amador  
**Carrera:** Ingeniería en Inteligencia Artificial · 7° semestre  
**Universidad:** Universidad Marista de Guadalajara  
**Materia:** Visión Artificial  
**Fecha:** 13 de noviembre de 2026

## Prácticas

| Práctica | Scripts | Reporte |
|---|---|---|
| 1 · Procesamiento de imágenes | `practica01/p1_paso1_cargar_inspeccionar.py`, `practica01/p1_paso2_resolucion_bits.py`, `practica01/p1_paso3_tipos_canales.py` | [reporte_practica01.pdf](practica01/reporte_practica01.pdf) (3 páginas) |
| 2 · Adquisición de video y filtrado | `practica02/p2_paso1_video.py`, `practica02/p2_paso2_suavizado.py`, `practica02/p2_paso3_realce.py` | [reporte_practica02.pdf](practica02/reporte_practica02.pdf) (4 páginas) |
| 3 · Segmentación | `practica03/p3_paso1_umbral.py` a `practica03/p3_paso7_comparativa.py` | [reporte_practica03.pdf](practica03/reporte_practica03.pdf) (6 páginas) + [anexo webcam](practica03/anexo_webcam_practica03.pdf) (1 página) |

Las capturas de la webcam están en `practica0N/capturas_webcam/` y las del video sintético en
`practica0N/capturas/`. En la Práctica 3 los scripts se importan entre sí, así que hay que ejecutarlos desde
`practica03/` o con esa carpeta en el `sys.path`.

## Cómo ejecutar

```bash
cd practica01
python -m venv va
source va/bin/activate          # Windows: va\Scripts\activate
pip install -r requirements.txt
python prueba_entorno.py
python p1_paso1_cargar_inspeccionar.py
python p1_paso2_resolucion_bits.py
python p1_paso3_tipos_canales.py
```

La Práctica 1 usa por defecto la foto `recursos/webcam.png`. Para capturar otra foto con la cámara:
`python p1_paso1_cargar_inspeccionar.py --camara`.

Para la Práctica 2 uso el mismo entorno de la Práctica 1 y el video grabado con mi webcam:

```bash
cd practica02
source ../practica01/va/bin/activate
python p2_paso1_video.py --video recursos/webcam.mp4
python p2_paso2_suavizado.py --video recursos/webcam.mp4
python p2_paso3_realce.py --video recursos/webcam.mp4
```

La Práctica 3, en orden (el paso 7 importa a los anteriores):

```bash
cd practica03
source ../practica01/va/bin/activate
python p3_paso1_umbral.py --video recursos/video_sintetico.mp4
python p3_paso2_bordes.py --video recursos/video_sintetico.mp4
python p3_paso3_agrupacion.py --video recursos/video_sintetico.mp4     # clic sobre la pelota
python p3_paso4_dominio.py --video recursos/video_sintetico.mp4        # clic sobre la pelota
python p3_paso5_movimiento.py --video recursos/video_sintetico.mp4
python p3_paso6_hough_snake.py --video recursos/video_sintetico.mp4    # clic en el centro de la pelota
python p3_paso7_comparativa.py                                          # tabla de IoU contra mascaras.npz
```

Para repetir los pasos 1, 2, 3 y 6 con mi webcam, cambio `--video recursos/video_sintetico.mp4` por
`--video recursos/webcam.mp4`. En los pasos con barras muevo los deslizadores y salgo con **Esc** o **Q**; en los pasos
con clic, hago clic dentro de la ventana indicada.

## Entorno con el que trabajé

- Python 3.14.8
- OpenCV 5.0.0
- NumPy 2.5.4
- Matplotlib 3.11.2
- scikit-image 0.26.0

## Nota

La Práctica 1 se hizo con una foto tomada con mi webcam (`recursos/webcam.png`) y la Práctica 2 con un video de 15 s
grabado con la misma cámara (`recursos/webcam.mp4`). En la Práctica 3 usé el video sintético para las preguntas y la
tabla de IoU, porque es el único que trae `mascaras.npz`; además repetí los pasos 1, 2, 3 y 6 con mi webcam y las
capturas están en el anexo.
