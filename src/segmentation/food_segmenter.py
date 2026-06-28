import cv2
import logging
import numpy as np
import os
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FoodSegmenter")

class FoodSegmenter:
    def __init__(self, model_path=settings.MODEL_PATH, conf=settings.CONFIDENCE_THRESHOLD, device=settings.DEVICE):
        self.model_path = model_path
        self.conf = conf
        self.device = device
        self.model = None

    def load_model(self):
        """Carga el modelo YOLO11 Segment."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No se encontró el archivo del modelo YOLO11 (modelofinal.pt) en '{self.model_path}'. "
                "Asegúrate de colocar el archivo entrenado en la raíz del proyecto."
            )

        try:
            from ultralytics import YOLO
            logger.info(f"Cargando modelo YOLO11 segment desde '{self.model_path}' en '{self.device}'...")
            self.model = YOLO(self.model_path)
            logger.info("Modelo YOLO11 segment cargado exitosamente.")
            return True
        except Exception as e:
            logger.error(f"Error al cargar el modelo YOLO11 con Ultralytics: {e}")
            raise e

    def segment(self, image):
        """
        Segmenta los alimentos en la imagen usando el modelo YOLO11.
        Retorna un diccionario estructurado:
        {
            'clase': {
                'area_px': total_pixeles_ocupados,
                'masks': list[numpy_binary_masks],
                'percentage': porcentaje_ocupacion
            }
        }
        Y la imagen procesada con las máscaras superpuestas.
        """
        if self.model is None:
            raise RuntimeError("El modelo no ha sido cargado. Llama a load_model() primero.")

        # Realizar inferencia
        results = self.model(image, conf=self.conf, device=self.device)[0]
        
        # Inicializar resultados
        detections = {}
        annotated_img = image.copy()
        height, width = image.shape[:2]
        total_tray_pixels = height * width
        
        # Obtener nombres de clases del modelo
        class_names = self.model.names
        
        if results.masks is not None:
            # Dibujar máscaras anotadas de YOLO
            annotated_img = results.plot(boxes=True, masks=True, labels=True)
            
            # Iterar sobre cada detección y sus máscaras correspondientes
            for mask_data, box in zip(results.masks.data, results.boxes):
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]
                
                # Convertir máscara de GPU tensor a numpy array y redimensionar al tamaño original
                mask_np = mask_data.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                area_px = int(np.sum(mask_binary))
                
                if class_name not in detections:
                    detections[class_name] = {
                        "area_px": 0,
                        "masks": [],
                        "percentage": 0.0
                    }
                
                detections[class_name]["masks"].append(mask_binary)
                detections[class_name]["area_px"] += area_px

            # Calcular porcentajes relativos
            for class_name, data in detections.items():
                data["percentage"] = (data["area_px"] / total_tray_pixels) * 100.0

        return detections, annotated_img
