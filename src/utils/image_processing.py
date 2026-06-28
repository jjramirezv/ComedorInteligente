import cv2
import numpy as np
import os

def overlay_mask(image, mask, color, alpha=0.4):
    """
    Superpone una máscara binaria de color sobre la imagen con una transparencia dada (alpha).
    """
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Combinar la imagen original con la máscara de color usando la transparencia
    masked_indices = mask > 0
    image[masked_indices] = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)[masked_indices]
    return image

def draw_annotations(image, detections, estimation_results):
    """
    Dibuja cuadros, contornos y etiquetas detalladas sobre la imagen,
    mostrando el tipo de alimento, el porcentaje de peso y el gramaje estimado.
    """
    annotated = image.copy()
    
    # Paleta de colores Premium para anotaciones (BGR)
    color_palette = {
        "nada": (128, 128, 128),          # Gris
        "arroz": (245, 245, 245),         # Blanco
        "mandarina": (0, 140, 255),       # Naranja oscuro
        "pescado frito": (30, 80, 180),   # Marrón cobrizo
        "yuca": (160, 220, 240)           # Amarillo crema / beige
    }
    default_color = (0, 165, 255) # Naranja por defecto

    # 1. Dibujar las máscaras coloreadas
    for class_name, data in detections.items():
        color = color_palette.get(class_name, default_color)
        for mask in data.get("masks", []):
            annotated = overlay_mask(annotated, mask, color, alpha=0.45)
            
            # Dibujar el contorno exterior con un trazo fino para definición
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2)

    # 2. Dibujar tarjetas de información (HUD) en la parte lateral/superior de la pantalla
    # Crear un fondo oscuro semitransparente para las métricas generales (tarjeta HUD)
    hud_bg = np.zeros_like(annotated)
    cv2.rectangle(hud_bg, (15, 15), (380, 280), (30, 30, 30), -1)
    # Mezclar el HUD oscuro sobre la imagen
    annotated = cv2.addWeighted(annotated, 0.85, hud_bg, 0.15, 0)
    
    # Borde de la tarjeta HUD
    cv2.rectangle(annotated, (15, 15), (380, 280), (100, 100, 100), 2)
    cv2.putText(annotated, "Métricas en Tiempo Real (UNCP)", (30, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.line(annotated, (30, 55), (365, 55), (100, 100, 100), 1)

    y_offset = 85
    for class_name, est_data in estimation_results.items():
        # Excluir la clave de debug no clasificada si existe
        if class_name == "desconocido_sin_segmentar":
            continue
            
        color = color_palette.get(class_name, default_color)
        weight_g = est_data["estimated_weight_g"]
        percent = est_data["percentage_weight"]
        
        # Dibujar una pequeña caja de color indicadora
        cv2.rectangle(annotated, (30, y_offset - 12), (45, y_offset), color, -1)
        cv2.rectangle(annotated, (30, y_offset - 12), (45, y_offset), (255, 255, 255), 1)
        
        # Texto del residuo
        text_label = f"{class_name.replace('_', ' ').capitalize()}"
        text_value = f"{weight_g:.1f}g ({percent:.1f}%)"
        
        cv2.putText(annotated, text_label, (55, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, text_value, (220, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        y_offset += 35

    # Dibujar indicador de peso total en la esquina inferior izquierda
    cv2.rectangle(annotated, (15, 660), (300, 705), (30, 30, 30), -1)
    cv2.rectangle(annotated, (15, 660), (300, 705), (100, 100, 100), 2)
    
    total_weight = sum([d["estimated_weight_g"] for d in estimation_results.values()])
    cv2.putText(annotated, f"PESO TOTAL REAL: {total_weight:.1f} g", (30, 690), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated

def save_frame(image, prefix="estimacion", output_dir="captures"):
    """
    Guarda el frame en un directorio específico con un timestamp para registro histórico.
    """
    import datetime
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, image)
    return path
