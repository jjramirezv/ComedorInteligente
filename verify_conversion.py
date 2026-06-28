import os
import glob

def load_simple_yaml(yaml_path):
    # Un parser manual simple de YAML para evitar dependencias
    content = {}
    current_key = None
    in_names = False
    names_dict = {}
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith('#'):
                continue
            
            # Controlar sangrado para "names"
            if line.startswith('  ') and in_names:
                parts = line_str.split(':', 1)
                if len(parts) == 2:
                    k = int(parts[0].strip())
                    v = parts[1].strip().strip("'").strip('"')
                    names_dict[k] = v
                continue
            else:
                in_names = False
                
            parts = line_str.split(':', 1)
            if len(parts) == 2:
                k = parts[0].strip()
                v = parts[1].strip()
                if k == 'names':
                    in_names = True
                else:
                    content[k] = v.strip("'").strip('"')
                    
    if names_dict:
        content['names'] = names_dict
    return content

def main():
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default original
    yolo_dataset_name = "Food types.v3i.yolo-segmentation"
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.endswith(".coco-segmentation"):
            yolo_dataset_name = arg.replace(".coco-segmentation", ".yolo-segmentation")
        elif arg.endswith(".yolo-segmentation"):
            yolo_dataset_name = arg
        else:
            yolo_dataset_name = arg + "-yolo" if not arg.endswith("-yolo") else arg
            
    dataset_dir = os.path.join(base_dir, yolo_dataset_name)
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    
    print("=== Iniciando Verificación de Conversión ===")
    if not os.path.exists(data_yaml_path):
        print(f"Error: No se encontró data.yaml en {data_yaml_path}")
        return
        
    yaml_data = load_simple_yaml(data_yaml_path)
    print("\nContenido de data.yaml cargado:")
    for k, v in yaml_data.items():
        if k == 'names':
            print(f"  names: {len(v)} clases registradas.")
        else:
            print(f"  {k}: {v}")
            
    num_classes = len(yaml_data.get('names', {}))
    # Determinar qué splits se van a verificar basándose en las carpetas existentes
    splits = []
    overall_valid = True
    
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.exists(split_dir):
            splits.append(split)
            
    if "train" not in splits:
        print("  Error: No se encontró la partición obligatoria 'train'.")
        overall_valid = False
        
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        
        print(f"\nVerificando partición '{split}':")
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"  Error: Faltan carpetas de imágenes o etiquetas para {split}")
            overall_valid = False
            continue
            
        images = glob.glob(os.path.join(images_dir, "*"))
        labels = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        print(f"  Imágenes físicas encontradas: {len(images)}")
        print(f"  Archivos de etiquetas (.txt): {len(labels)}")
        
        if len(images) != len(labels):
            print(f"  Advertencia: Desajuste entre imágenes ({len(images)}) y etiquetas ({len(labels)}).")
            # En COCO a veces hay imágenes sin anotaciones. El convertidor genera archivos txt vacíos,
            # por lo que deberían ser iguales a menos que hubiera imágenes faltantes en disco.
            
        # Validar una muestra de archivos de etiquetas
        error_count = 0
        total_polygons = 0
        
        for lf_path in labels[:50]:  # Validar hasta 50 archivos de etiquetas por split
            with open(lf_path, 'r', encoding='utf-8') as lf:
                lines = lf.readlines()
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                total_polygons += 1
                try:
                    class_idx = int(parts[0])
                    if class_idx < 0 or class_idx >= num_classes:
                        print(f"  [Error] {os.path.basename(lf_path)} L{line_idx+1}: Índice de clase {class_idx} fuera de rango.")
                        error_count += 1
                    
                    # Validar coordenadas (deben ser pares)
                    coords = parts[1:]
                    if len(coords) < 6 or len(coords) % 2 != 0:
                        print(f"  [Error] {os.path.basename(lf_path)} L{line_idx+1}: Coordenadas inválidas (cantidad impar o insuficiente: {len(coords)}).")
                        error_count += 1
                        
                    for val_str in coords:
                        val = float(val_str)
                        if val < 0.0 or val > 1.0:
                            print(f"  [Error] {os.path.basename(lf_path)} L{line_idx+1}: Coordenada {val} fuera del rango [0.0, 1.0].")
                            error_count += 1
                except ValueError as e:
                    print(f"  [Error] {os.path.basename(lf_path)} L{line_idx+1}: Error al parsear valores. {e}")
                    error_count += 1
                    
        if error_count > 0:
            print(f"  Resultado de muestra: {error_count} errores detectados en validación de formato.")
            overall_valid = False
        else:
            print(f"  Resultado de muestra: Formato de etiquetas 100% válido ({total_polygons} polígonos verificados).")
            
    if overall_valid:
        print("\n=== ¡VERIFICACIÓN EXITOSA! El dataset está listo para YOLO segment. ===")
    else:
        print("\n=== HOB HOB... Se encontraron algunos errores en el dataset verificado. ===")

if __name__ == "__main__":
    main()
