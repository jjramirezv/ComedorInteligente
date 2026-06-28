import cv2
import logging
import numpy as np
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("USBCamera")

class USBCamera:
    def __init__(self, index=settings.CAMERA_INDEX, width=settings.CAMERA_WIDTH, height=settings.CAMERA_HEIGHT):
        self.index = index
        self.width = width
        self.height = height
        self.cap = None

    def initialize(self):
        """Inicializa la captura de video desde la cámara web USB."""
        logger.info(f"Intentando inicializar cámara en índice {self.index}...")
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW if os_name() == "nt" else cv2.CAP_ANY)
        
        if not self.cap.isOpened():
            logger.warning(f"No se pudo abrir la cámara física en el índice {self.index}.")
            self.cap = None
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info(f"Cámara física inicializada con resolución {self.width}x{self.height}.")
        return True

    def capture_frame(self, fallback_path=None):
        """
        Captura un frame de la cámara.
        Si la cámara física no está disponible, intenta cargar una imagen de prueba
        o genera un patrón sintético de prueba.
        """
        if self.cap is not None:
            # Descartar los primeros frames para permitir autoexposición
            for _ in range(5):
                self.cap.read()
            ret, frame = self.cap.read()
            if ret:
                logger.info("Frame capturado exitosamente de la cámara física.")
                return frame
            else:
                logger.error("Error al leer de la cámara física.")
        
        # --- Fallback si no hay cámara física ---
        if fallback_path and cv2.os.path.exists(fallback_path):
            logger.info(f"Cámara física no disponible. Cargando imagen de prueba desde {fallback_path}")
            frame = cv2.imread(fallback_path)
            if frame is not None:
                return frame

        # Generar imagen sintética de bandeja para pruebas sin hardware
        logger.info("Cámara física no disponible. Generando frame sintético de bandeja de metal...")
        return self._generate_synthetic_tray()

    def _generate_synthetic_tray(self):
        """Genera una imagen que simula una bandeja metálica con compartimentos para pruebas."""
        # Imagen de 1280x720 (fondo gris plateado que simula metal)
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 180
        
        # Dibujar líneas de compartimentos para simular relieve
        # Compartimento principal de comida (abajo)
        cv2.rectangle(img, (100, 350), (1180, 680), (120, 120, 120), 4) # Borde
        # Compartimento superior izquierdo (bananas / ensalada)
        cv2.rectangle(img, (100, 50), (600, 300), (120, 120, 120), 4)
        # Compartimento superior derecho (sopa / postre)
        cv2.rectangle(img, (680, 50), (1180, 300), (120, 120, 120), 4)
        
        # Agregar ruido visual para simular reflectividad de metal y textura
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Simular algunos alimentos como círculos/polígonos de colores para validación
        # Arroz (Círculo blanco en el compartimento principal)
        cv2.circle(img, (350, 510), 120, (240, 240, 240), -1)
        # Carne/Pollo (Polígono marrón)
        pts_carne = np.array([[600, 420], [750, 400], [800, 520], [620, 540]], np.int32)
        cv2.fillPoly(img, [pts_carne], (45, 82, 139))
        # Papas (Círculos amarillos)
        cv2.circle(img, (820, 580), 40, (100, 200, 220), -1)
        cv2.circle(img, (900, 530), 45, (100, 200, 220), -1)
        # Verduras (Polígono verde en el compartimento superior izquierdo)
        pts_ensalada = np.array([[150, 100], [400, 80], [450, 250], [180, 220]], np.int32)
        cv2.fillPoly(img, [pts_ensalada], (60, 160, 60))
        # Huesos de residuo (Círculo beige claro)
        cv2.circle(img, (1050, 450), 30, (200, 225, 240), -1)
        
        cv2.putText(img, "SIMULADOR CAMARA (BANDEJA UNCP)", (400, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        return img

    def release(self):
        """Libera la cámara física."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Cámara física liberada.")

def os_name():
    import platform
    return platform.system().lower()
