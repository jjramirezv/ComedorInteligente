# Sistema de Visión Artificial para Estimación de Desperdicios Alimentarios (UNCP)

Este proyecto piloto utiliza procesamiento de imágenes y algoritmos de estimación ponderada por densidad de alimentos para calcular el gramaje individual de los desperdicios en las bandejas de retorno del comedor de la UNCP.

El sistema está optimizado para funcionar en un **Raspberry Pi 5 (8GB RAM)**.

---

## Estructura del Proyecto

*   `config/settings.py`: Parámetros de la cámara web, balanza serial, factores de densidad e inferencia.
*   `src/hardware/camera.py`: Módulo de captura cenital. Si no hay cámara física conectada, simula visualmente la bandeja con los alimentos básicos.
*   `src/hardware/scale.py`: Módulo de conexión con balanza serial (PySerial). Contiene `MockScale` interactivo si no está conectada la balanza física.
*   `src/segmentation/food_segmenter.py`: Ejecuta la segmentación con **YOLO11 Segment** (`yolo11n-seg` para Raspberry Pi). En su ausencia, realiza segmentación simulada por color.
*   `src/estimation/weight_estimator.py`: Algoritmo que calcula el peso de cada alimento multiplicando área por densidad y normalizándolo al peso de la balanza.
*   `src/utils/image_processing.py`: Herramientas gráficas OpenCV para superponer máscaras de colores y dibujar un panel HUD de métricas.
*   `main.py`: Script principal e interactivo.
*   `tests/test_pipeline.py`: Pruebas unitarias de la lógica del sistema.

---

## Requisitos e Instalación

1. Instalar las dependencias de Python:
   ```bash
   pip install -r requirements.txt
   ```

2. (Opcional) Descargar o entrenar el modelo de YOLO11 segment:
   * Coloque su archivo de modelo entrenado de Roboflow en el directorio raíz del proyecto con el nombre `yolo11n-seg.pt` (o actualice la ruta en `config/settings.py`).

---

## Instrucciones de Ejecución

### 1. Ejecutar el Script Principal
Inicie el programa con:
```bash
python main.py
```

*   **Sin hardware físico (Modo Simulación):** El programa cargará una bandeja cenital simulada y solicitará en la terminal el peso en gramos registrado por la balanza.
*   **Con cámara USB física:** Se abrirá una ventana de calibración con el feed de video continuo. Presione **ESPACIO** o **ENTER** para congelar el cuadro, registrar el peso y estimar los desperdicios.
*   **Guardar Resultados:** El programa guarda automáticamente la captura original y el análisis con máscaras de color en una carpeta local llamada `captures/`.

### 2. Ejecutar Pruebas Automatizadas
Para validar que los cálculos matemáticos, estimación y utilidades funcionan de forma idéntica en cualquier sistema:
```bash
python -m unittest tests/test_pipeline.py
```

---

## Algoritmo de Estimación Utilizado

Dado que diferentes alimentos tienen densidades muy distintas (ej. los huesos de pollo/carne pesan más por centímetro cuadrado que la lechuga), el peso de cada categoría se estima ponderando el área pixelar ($A_i$) con su factor de densidad ($D_i$):

$$Volumen\_Estimado_i = A_i \times D_i$$

$$Peso\_Estimado_i = Peso\_Balanza \times \frac{Volumen\_Estimado_i}{\sum Volumen\_Estimado_k}$$

Las densidades por defecto se encuentran en `config/settings.py` y pueden ajustarse mediante calibraciones de pesaje real:
*   Arroz: `1.0`
*   Carne/Pollo: `1.4`
*   Papas/Segundo: `1.1`
*   Ensalada/Verduras: `0.6`
*   Huesos/Desechos: `1.5`
