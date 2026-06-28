import os
import json
import shutil
import yaml

def decode_rle_counts(data: str):
    m = len(data)
    counts = []
    p = 0
    while p < m:
        x = 0
        k = 0
        more = 1
        while more > 0:
            c = ord(data[p]) - 48
            x |= (c & 0x1F) << 5 * k
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << 5 * k
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts

def main():
    # Rutas relativas basadas en el directorio de trabajo (raíz del proyecto)
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default original
    coco_dataset_name = "jhondemrd.v1i.coco-segmentation"
    
    # Si se pasa un argumento por línea de comandos, usarlo
    if len(sys.argv) > 1:
        coco_dataset_name = sys.argv[1]
        
    if coco_dataset_name.endswith(".coco-segmentation"):
        yolo_dataset_name = coco_dataset_name.replace(".coco-segmentation", ".yolo-segmentation")
    else:
        yolo_dataset_name = coco_dataset_name + "-yolo"
    
    src_dataset_dir = os.path.join(base_dir, coco_dataset_name)
    dst_dataset_dir = os.path.join(base_dir, yolo_dataset_name)
    
    splits = ["train", "valid", "test"]
    
    print(f"Buscando dataset COCO en: {src_dataset_dir}")
    if not os.path.exists(src_dataset_dir):
        print(f"Error: No se encontró la carpeta de origen {src_dataset_dir}")
        return
        
    # 1. Obtener todas las categorías para construir un mapeo unificado y consistente
    print("Escaneando categorías en todos los splits...")
    all_categories = {}
    for split in splits:
        json_path = os.path.join(src_dataset_dir, split, "_annotations.coco.json")
        if not os.path.exists(json_path):
            print(f"Advertencia: No se encontró annotations en {json_path}")
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for cat in data.get('categories', []):
            cat_id = cat['id']
            cat_name = cat['name']
            all_categories[cat_id] = cat_name
            
    if not all_categories:
        print("Error: No se encontraron categorías en los archivos de anotaciones.")
        return
        
    # Mapear IDs de categoría COCO originales a índices continuos de YOLO [0 ... N-1]
    sorted_coco_ids = sorted(all_categories.keys())
    class_map = {coco_id: idx for idx, coco_id in enumerate(sorted_coco_ids)}
    yolo_names = {idx: all_categories[coco_id] for idx, coco_id in enumerate(sorted_coco_ids)}
    
    print(f"Total de categorías detectadas: {len(yolo_names)}")
    for idx, name in yolo_names.items():
        print(f"  Clase {idx}: {name}")
        
    # Crear estructura del nuevo dataset YOLO
    print(f"\nCreando la estructura del nuevo dataset YOLO en: {dst_dataset_dir}")
    os.makedirs(dst_dataset_dir, exist_ok=True)
    
    # 2. Procesar cada split
    for split in splits:
        split_src_dir = os.path.join(src_dataset_dir, split)
        
        json_path = os.path.join(split_src_dir, "_annotations.coco.json")
        if not os.path.exists(json_path):
            print(f"Saltando split {split} por falta de annotations.")
            continue
            
        split_dst_images_dir = os.path.join(dst_dataset_dir, split, "images")
        split_dst_labels_dir = os.path.join(dst_dataset_dir, split, "labels")
        
        os.makedirs(split_dst_images_dir, exist_ok=True)
        os.makedirs(split_dst_labels_dir, exist_ok=True)
            
        print(f"\nProcesando split '{split}'...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        
        # Mapear image_id a info de la imagen
        image_dict = {img['id']: img for img in images}
        
        # Agrupar anotaciones por image_id
        annotations_by_img = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in annotations_by_img:
                annotations_by_img[img_id] = []
            annotations_by_img[img_id].append(ann)
            
        success_count = 0
        skip_count = 0
        
        for img_id, img_info in image_dict.items():
            file_name = img_info['file_name']
            width = img_info['width']
            height = img_info['height']
            
            src_image_path = os.path.join(split_src_dir, file_name)
            if not os.path.exists(src_image_path):
                print(f"Advertencia: No se encontró la imagen física {src_image_path}. Saltando.")
                skip_count += 1
                continue
                
            # Copiar imagen a la nueva carpeta
            dst_image_path = os.path.join(split_dst_images_dir, file_name)
            shutil.copy2(src_image_path, dst_image_path)
            
            # Generar archivo de etiquetas txt correspondiente
            base_name = os.path.splitext(file_name)[0]
            label_file_path = os.path.join(split_dst_labels_dir, f"{base_name}.txt")
            
            img_anns = annotations_by_img.get(img_id, [])
            
            with open(label_file_path, 'w', encoding='utf-8') as lf:
                for ann in img_anns:
                    coco_cat_id = ann['category_id']
                    if coco_cat_id not in class_map:
                        continue
                    yolo_class_idx = class_map[coco_cat_id]
                    
                    segmentations = ann.get('segmentation', [])
                    polygons = []
                    
                    if isinstance(segmentations, list):
                        # Formato de Polígonos estándar
                        for seg in segmentations:
                            if len(seg) >= 6:
                                polygons.append(seg)
                    elif isinstance(segmentations, dict):
                        # Formato RLE (Run-Length Encoding)
                        counts_data = segmentations.get('counts')
                        size_data = segmentations.get('size')
                        if counts_data and size_data:
                            h, w = size_data[0], size_data[1]
                            try:
                                # Decodificar RLE a máscara binaria
                                if isinstance(counts_data, str):
                                    counts = decode_rle_counts(counts_data)
                                else:
                                    counts = counts_data
                                    
                                import numpy as np
                                import cv2
                                
                                # Reconstruir máscara 1D
                                mask_1d = np.zeros(h * w, dtype=np.uint8)
                                current_val = 0
                                current_idx = 0
                                for count in counts:
                                    if count > 0:
                                        mask_1d[current_idx : current_idx + count] = current_val
                                        current_idx += count
                                    current_val = 1 - current_val
                                    
                                # Reestructurar en 2D (F-order para COCO RLE)
                                binary_mask = mask_1d.reshape((h, w), order='F')
                                
                                # Encontrar contornos/polígonos usando OpenCV
                                contours, _ = cv2.findContours(binary_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for contour in contours:
                                    if len(contour) >= 3:
                                        poly = contour.reshape(-1).tolist()
                                        polygons.append(poly)
                            except Exception as e:
                                print(f"Error decodificando RLE para anotación {ann.get('id')}: {e}")
                                
                    # Escribir polígonos a archivo YOLO
                    for seg in polygons:
                        normalized_coords = []
                        for i in range(0, len(seg), 2):
                            x = seg[i] / width
                            y = seg[i+1] / height
                            
                            # Asegurar que queden en el rango [0.0, 1.0]
                            x = max(0.0, min(1.0, x))
                            y = max(0.0, min(1.0, y))
                            normalized_coords.append(f"{x:.6f} {y:.6f}")
                            
                        lf.write(f"{yolo_class_idx} " + " ".join(normalized_coords) + "\n")
            
            success_count += 1
            
        print(f"Split '{split}' completado. {success_count} imágenes procesadas con éxito, {skip_count} omitidas.")

    # 3. Crear archivo data.yaml
    data_yaml_path = os.path.join(dst_dataset_dir, "data.yaml")
    
    # Determinar qué splits existen realmente para configurar data.yaml de forma robusta
    train_path = 'train/images'
    val_path = 'valid/images' if os.path.exists(os.path.join(dst_dataset_dir, 'valid', 'images')) else 'train/images'
    test_path = 'test/images' if os.path.exists(os.path.join(dst_dataset_dir, 'test', 'images')) else None
    
    # Generar rutas relativas adecuadas para data.yaml (usando "/" como convención YOLO)
    yaml_content = {
        'path': yolo_dataset_name,  # Ruta relativa al directorio de ejecución
        'train': train_path,
        'val': val_path
    }
    if test_path:
        yaml_content['test'] = test_path
        
    yaml_content['names'] = yolo_names
    
    try:
        import yaml
        with open(data_yaml_path, 'w', encoding='utf-8') as yf:
            yaml.dump(yaml_content, yf, default_flow_style=False, sort_keys=False)
        print("data.yaml generado usando PyYAML.")
    except ImportError:
        # Fallback manual por si PyYAML no está instalado
        with open(data_yaml_path, 'w', encoding='utf-8') as yf:
            yf.write(f"path: {yaml_content['path']}\n")
            yf.write(f"train: {yaml_content['train']}\n")
            yf.write(f"val: {yaml_content['val']}\n")
            yf.write(f"test: {yaml_content['test']}\n\n")
            yf.write("names:\n")
            for idx, name in yaml_content['names'].items():
                # Escapar nombres por si contienen caracteres especiales
                yf.write(f"  {idx}: '{name}'\n")
        print("data.yaml generado manualmente (PyYAML no disponible).")
        
    print(f"\nArchivo data.yaml generado con éxito en: {data_yaml_path}")
    print("Conversión completada con éxito.")

if __name__ == "__main__":
    main()
