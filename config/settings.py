import os

#MODO DESARROLLO 
MOCK_MODE = True

#CONFIGURACIÓN cámara
CAMERA_INDEX = 0          # Índice de la cámara USB
CAMERA_WIDTH = 1280        # Ancho de la captura
CAMERA_HEIGHT = 720        # Alto de la captura

#CONFIGURACIÓN DE BALANZA FISICA 
SERIAL_PORT = "COM3"       
BAUD_RATE = 9600
TIMEOUT = 1.0


MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelofinal.pt")
CONFIDENCE_THRESHOLD = 0.25
DEVICE = "cpu"           

CLASS_DENSITIES = {
    "nada": 0.0,             # No comida / nada
    "arroz": 1.0,            # Arroz
    "mandarina": 0.8,        # Mandarina
    "pescado frito": 1.2,    # Pescado frito
    "yuca": 1.3              # Yuca
}

DEFAULT_DENSITY = 1.0         # Para cualquier clase no especificada
