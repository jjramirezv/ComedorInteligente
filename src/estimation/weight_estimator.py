import logging
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WeightEstimator")

class WeightEstimator:
    def __init__(self, densities=settings.CLASS_DENSITIES, default_density=settings.DEFAULT_DENSITY):
        self.densities = densities
        self.default_density = default_density

    def estimate_weights(self, detections, total_weight):
        """
        Calcula la estimación de peso para cada alimento detectado.
        
        detections: Diccionario retornado por el segmentador con la estructura:
                    { 'arroz': { 'area_px': 12345, ... } }
        total_weight: El peso real leído de la balanza (en gramos).
        
        Retorna:
            Diccionario con la distribución de peso estimado por clase.
            Estructura:
            {
                'arroz': {
                    'area_px': 12345,
                    'density': 1.0,
                    'estimated_volume': 12345.0,
                    'estimated_weight_g': 150.5,
                    'percentage_weight': 43.0
                },
                ...
            }
        """
        results = {}
        
        # Validar si no hay detecciones pero sí hay peso registrado
        if not detections:
            logger.warning("No se detectaron alimentos segmentados, pero la balanza registra peso.")
            if total_weight > 0:
                results["desconocido_sin_segmentar"] = {
                    "area_px": 0,
                    "density": 1.0,
                    "estimated_volume": 0.0,
                    "estimated_weight_g": total_weight,
                    "percentage_weight": 100.0
                }
            return results

        total_volume = 0.0
        
        # 1. Calcular volumen estimado para cada clase: V = Area * Densidad
        for class_name, data in detections.items():
            area = data["area_px"]
            density = self.densities.get(class_name, self.default_density)
            est_volume = area * density
            total_volume += est_volume
            
            results[class_name] = {
                "area_px": area,
                "density": density,
                "estimated_volume": est_volume,
                "estimated_weight_g": 0.0,
                "percentage_weight": 0.0
            }
        
        # 2. Distribuir el peso real en base a la proporción del volumen estimado
        if total_volume > 0:
            for class_name, res in results.items():
                proportion = res["estimated_volume"] / total_volume
                est_weight = total_weight * proportion
                res["estimated_weight_g"] = round(est_weight, 2)
                res["percentage_weight"] = round(proportion * 100, 2)
        else:
            # Fallback por si la suma de volúmenes es cero (evitar división por cero)
            num_classes = len(results)
            for class_name, res in results.items():
                est_weight = total_weight / num_classes
                res["estimated_weight_g"] = round(est_weight, 2)
                res["percentage_weight"] = round((1.0 / num_classes) * 100, 2)
                
        return results

    def print_results(self, results, total_weight):
        """Imprime un resumen formateado de la estimación en la terminal."""
        print("\n" + "="*50)
        print("          RESULTADO DE LA ESTIMACIÓN DE PESO          ")
        print("="*50)
        print(f"Peso Total Registrado por Balanza: {total_weight:.1f} g\n")
        print(f"{'Categoría':<20} | {'Área (px)':<10} | {'Densidad':<8} | {'Peso Est (g)':<12} | {'% Peso':<6}")
        print("-"*69)
        
        for name, data in results.items():
            print(f"{name:<20} | {data['area_px']:<10d} | {data['density']:<8.2f} | {data['estimated_weight_g']:<12.1f} | {data['percentage_weight']:.1f}%")
            
        print("="*50 + "\n")
