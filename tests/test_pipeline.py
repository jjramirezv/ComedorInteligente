import unittest
import numpy as np
import sys
import os

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.hardware.camera import USBCamera
from src.hardware.scale import DigitalScale
from src.segmentation.food_segmenter import FoodSegmenter
from src.estimation.weight_estimator import WeightEstimator
from src.utils.image_processing import draw_annotations

class TestUNCPWastePipeline(unittest.TestCase):
    
    def setUp(self):
        # Asegurar configuraciones consistentes para el test
        self.densities = {
            "Rice": 1.0,
            "Chicken": 1.4,
            "Bones": 1.5,
            "Potato": 1.1,
            "Green veg": 0.6
        }
        self.estimator = WeightEstimator(densities=self.densities, default_density=1.0)
        self.scale = DigitalScale()
        self.camera = USBCamera()
        self.segmenter = FoodSegmenter()
        # Mock de segmentación para evitar cargar YOLO de verdad en los tests
        self.segmenter.segment = self._mock_segment_method

    def _mock_segment_method(self, image):
        """Simulador de segmentación para tests unitarios."""
        height, width = image.shape[:2]
        total_pixels = height * width
        
        # Simular máscaras
        mask_rice = np.zeros((height, width), dtype=np.uint8)
        mask_rice[400:600, 200:500] = 1
        
        mask_chicken = np.zeros((height, width), dtype=np.uint8)
        mask_chicken[400:600, 600:800] = 1
        
        mask_potato = np.zeros((height, width), dtype=np.uint8)
        mask_potato[500:600, 800:1000] = 1
        
        mask_veg = np.zeros((height, width), dtype=np.uint8)
        mask_veg[100:250, 200:400] = 1
        
        mask_bones = np.zeros((height, width), dtype=np.uint8)
        mask_bones[400:500, 1000:1100] = 1
        
        detections = {
            "Rice": {
                "area_px": int(np.sum(mask_rice)),
                "masks": [mask_rice],
                "percentage": (np.sum(mask_rice) / total_pixels) * 100.0
            },
            "Chicken": {
                "area_px": int(np.sum(mask_chicken)),
                "masks": [mask_chicken],
                "percentage": (np.sum(mask_chicken) / total_pixels) * 100.0
            },
            "Potato": {
                "area_px": int(np.sum(mask_potato)),
                "masks": [mask_potato],
                "percentage": (np.sum(mask_potato) / total_pixels) * 100.0
            },
            "Green veg": {
                "area_px": int(np.sum(mask_veg)),
                "masks": [mask_veg],
                "percentage": (np.sum(mask_veg) / total_pixels) * 100.0
            },
            "Bones": {
                "area_px": int(np.sum(mask_bones)),
                "masks": [mask_bones],
                "percentage": (np.sum(mask_bones) / total_pixels) * 100.0
            }
        }
        
        annotated = image.copy()
        return detections, annotated

    def test_mock_scale_non_interactive(self):
        """Verifica que la balanza virtual retorne valores válidos en modo automático (no interactivo)."""
        # Forzar lectura no interactiva
        weight = self.scale._read_mock_weight(interactive=False)
        self.assertIsInstance(weight, float)
        self.assertTrue(100.0 <= weight <= 600.0)

    def test_mock_scale_with_preset_weight(self):
        """Verifica que read_weight() retorne el peso provisto en preset_weight si es válido."""
        weight = self.scale.read_weight(preset_weight="425.5")
        self.assertEqual(weight, 425.5)
        
        # Test con un valor no válido (debería cambiar a mock)
        weight_fallback = self.scale.read_weight(preset_weight="abc")
        self.assertIsInstance(weight_fallback, float)

    def test_density_weight_estimation_math(self):
        """
        Verifica el algoritmo matemático de estimación ponderada por densidad.
        Caso de prueba:
          - Rice: área = 10000 px, densidad = 1.0  => Vol = 10000
          - Chicken: área = 10000 px, densidad = 1.4  => Vol = 14000
          - Volumen Total = 24000
          - Peso Total = 240g
          - Peso Esperado Rice = 240 * (10000 / 24000) = 100g
          - Peso Esperado Chicken = 240 * (14000 / 24000) = 140g
        """
        detections = {
            "Rice": {
                "area_px": 10000,
                "masks": [np.ones((100, 100), dtype=np.uint8)],
                "percentage": 10.0
            },
            "Chicken": {
                "area_px": 10000,
                "masks": [np.ones((100, 100), dtype=np.uint8)],
                "percentage": 10.0
            }
        }
        
        total_weight = 240.0
        results = self.estimator.estimate_weights(detections, total_weight)
        
        # Verificar asignación de pesos
        self.assertIn("Rice", results)
        self.assertIn("Chicken", results)
        
        self.assertEqual(results["Rice"]["estimated_weight_g"], 100.0)
        self.assertEqual(results["Chicken"]["estimated_weight_g"], 140.0)
        self.assertEqual(results["Rice"]["percentage_weight"], 41.67)
        self.assertEqual(results["Chicken"]["percentage_weight"], 58.33)
        
        # Validar la suma de pesos estimados
        sum_weights = sum([data["estimated_weight_g"] for data in results.values()])
        self.assertAlmostEqual(sum_weights, total_weight, places=1)

    def test_mock_image_segmentation(self):
        """Verifica que el segmentador simule correctamente la detección de las 5 categorías básicas."""
        mock_frame = self.camera._generate_synthetic_tray()
        self.assertIsNotNone(mock_frame)
        self.assertEqual(mock_frame.shape, (720, 1280, 3))
        
        detections, annotated = self.segmenter.segment(mock_frame)
        
        # Deben detectarse las categorías dibujadas en la imagen sintética
        expected_classes = ["Rice", "Chicken", "Potato", "Green veg", "Bones"]
        for cls in expected_classes:
            self.assertIn(cls, detections, f"Clase {cls} faltante en detección mock.")
            self.assertGreater(detections[cls]["area_px"], 0, f"Área de {cls} debe ser mayor que 0.")
            self.assertTrue(len(detections[cls]["masks"]) > 0)

    def test_visualization_drawing(self):
        """Verifica que las funciones de anotación dibujen sobre el lienzo sin fallos."""
        mock_frame = self.camera._generate_synthetic_tray()
        detections, _ = self.segmenter.segment(mock_frame)
        
        estimation_results = {
            "Rice": {"area_px": 5000, "estimated_weight_g": 100.0, "percentage_weight": 50.0},
            "Chicken": {"area_px": 2000, "estimated_weight_g": 80.0, "percentage_weight": 40.0},
            "Bones": {"area_px": 500, "estimated_weight_g": 20.0, "percentage_weight": 10.0}
        }
        
        annotated_hud = draw_annotations(mock_frame, detections, estimation_results)
        self.assertEqual(annotated_hud.shape, mock_frame.shape)

if __name__ == "__main__":
    unittest.main()
