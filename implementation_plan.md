# Plan de Implementación: Interfaz Gráfica (GUI) para Estimación de Desperdicios (UNCP)

Este plan detalla la migración del script principal en modo consola/OpenCV a una **interfaz gráfica completa en Tkinter** compatible con Windows y Raspberry Pi 5. La interfaz permitirá ver el feed de la cámara web en tiempo real, seleccionar imágenes locales para procesar, ingresar el peso de la balanza digital directamente en la interfaz (modo simulado) y visualizar el resultado analizado con máscaras y HUD informativo.

---

## User Review Required

> [!IMPORTANT]
> **Dependencia de Pillow:** Se requiere instalar la librería `pillow` (PIL) para poder renderizar las imágenes y el flujo de video en los widgets de Tkinter. Se actualizará `requirements.txt`.
>
> **Entrada del Peso en GUI:** En lugar de solicitar el peso por consola en modo simulado (lo cual bloquearía la GUI), se agregará un campo de entrada de texto (Entry) en la misma ventana de la interfaz gráfica para que el usuario digite el peso de las sobras antes de presionar "Estimar". Si el puerto serial de la balanza física está conectado y activo, este campo se auto-completará con la lectura en tiempo real.

---

## Proposed Changes

### 1. Actualización de Requisitos e Integración de Hardware
*   **[requirements.txt](file:///c:/Users/ACER/Documents/chamba%20xd/iConocimiento/requirements.txt):** Agregar `pillow>=10.0.0` para el manejo de imágenes en Tkinter.
*   **[scale.py](file:///c:/Users/ACER/Documents/chamba%20xd/iConocimiento/src/hardware/scale.py):** Ajustar la función de lectura de peso mock para evitar bloquear la interfaz. Si no es interactiva por terminal, retornará el peso provisto por el campo de texto de la GUI.

### 2. Creación del Panel de Interfaz de Usuario (GUI)
*   **[main.py](file:///c:/Users/ACER/Documents/chamba%20xd/iConocimiento/main.py):** Reemplazar el bucle interactivo de consola por una aplicación de escritorio basada en `tkinter`.
    *   **Diseño Estético:** Fondo oscuro moderno (Dark Mode), tipografía limpia, botones interactivos con efectos de hover.
    *   **Panel Izquierdo:** Visualización de video en vivo (cámara web USB o feed simulado si no hay hardware).
    *   **Panel Derecho (Control y Resultados):**
        *   Campo para configurar/ver el Peso de la Balanza (entrada de texto modificable).
        *   Botón "Capturar y Estimar" (captura el cuadro de la cámara y procesa).
        *   Botón "Adjuntar Imagen Local" (abre un buscador de archivos para cargar una foto en formato JPG/PNG).
        *   Tabla o lista de resultados que muestre en tiempo real el gramaje y porcentaje de cada residuo detectado (`arroz`, `carne_pollo`, etc.).
    *   **Ventana de Resultados:** Al procesar, se mostrará en una sección de la interfaz (o en una ventana emergente premium) la imagen resultante con las máscaras coloreadas y el HUD con la estimación.

---

## Módulos a Modificar y Crear

#### [MODIFY] [requirements.txt](file:///c:/Users/ACER/Documents/chamba%20xd/iConocimiento/requirements.txt)
Agregar `pillow` a la lista de dependencias del proyecto.

#### [MODIFY] [scale.py](file:///c:/Users/ACER/Documents/chamba%20xd/iConocimiento/src/hardware/scale.py)
Agregar un parámetro opcional a `read_weight(preset_weight=None)` para que el hilo de interfaz de usuario pueda inyectar el peso digitado directamente en el GUI cuando se está en modo mock, evitando entradas por consola bloqueantes.

#### [MODIFY] [main.py](file:///c:/Users/ACER/Documents/chamba%20xd/iConocimiento/main.py)
Rediseñar el punto de entrada para levantar la ventana de Tkinter, gestionar el hilo de actualización de frames de la cámara USB y orquestar el procesamiento de imágenes subidas o capturadas.

---

## Verification Plan

### Automated & Unit Tests
Ajustar los test en `tests/test_pipeline.py` para asegurar que las modificaciones en `scale.py` sigan funcionando correctamente y no rompan la compatibilidad matemática.

### Manual Verification
*   **Prueba de Carga de Imagen:** Iniciar la GUI, hacer clic en "Adjuntar Imagen", seleccionar un archivo local de prueba y verificar que se realice la segmentación por color (o YOLO si está el modelo) y se calcule el peso estimado basándose en el valor de la balanza ingresado en la interfaz.
*   **Prueba de Cámara en Vivo:** Verificar que el feed de video se actualice fluidamente en la interfaz gráfica (o el patrón simulado de bandeja si no hay webcam).
