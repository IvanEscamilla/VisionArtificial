# Visión Artificial

Material de prácticas del curso de Visión Artificial (Lic. en Inteligencia Artificial, 7° semestre).

## Prácticas

| Carpeta | Práctica | Guía |
|---|---|---|
| `practica01/` | Procesamiento de imágenes: la imagen digital | [P01_Procesamiento_Imagenes.pdf](practica01/P01_Procesamiento_Imagenes.pdf) |
| `practica02/` | Adquisición de video y filtrado | [P02_Video_Filtrado.pdf](practica02/P02_Video_Filtrado.pdf) |
| `practica03/` | Segmentación | [P03_Segmentacion.pdf](practica03/P03_Segmentacion.pdf) |

## Estructura de cada práctica

Todas las prácticas tienen la misma estructura. Los scripts de cada paso (`p1_paso1_...py`, etc.) **los escribes tú** siguiendo la guía en PDF.

```text
practica01/
├── va/                                <- entorno virtual (lo creas tú, no está en el repo)
├── recursos/                          <- imágenes o videos de trabajo
├── requirements.txt                   <- librerías que hay que instalar
├── prueba_entorno.py                  <- comprueba que todo está instalado
├── P01_Procesamiento_Imagenes.pdf     <- guía de la práctica
├── p1_paso1_cargar_inspeccionar.py    <- los escribes tú
├── p1_paso2_resolucion_bits.py
└── p1_paso3_tipos_canales.py
```

## Cómo empezar

1. Descarga el repositorio (*Code → Download ZIP*) o clónalo:

   ```bash
   git clone https://github.com/IvanEscamilla/VisionArtificial.git
   ```

2. Entra a la carpeta de la práctica, crea y activa el entorno virtual e instala las librerías:

   ```bash
   cd VisionArtificial/practica01
   python -m venv va
   # Windows (PowerShell): va\Scripts\Activate.ps1
   # Windows (cmd):        va\Scripts\activate.bat
   # macOS / Linux:        source va/bin/activate
   pip install -r requirements.txt
   ```

3. Comprueba el entorno:

   ```bash
   python prueba_entorno.py
   ```

Requisitos: Python 3.10 o superior y VS Code con la extensión de Python. Una webcam es recomendable; si no tienes, las prácticas 2 y 3 funcionan con el video de `recursos/` usando `--video`.
