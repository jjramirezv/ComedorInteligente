import os
import sys
import argparse
import threading
from ultralytics import YOLO

# Intentar importar Tkinter para la interfaz gráfica
try:
    import tkinter as tk
    from tkinter import filedialog, scrolledtext, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO11 Segment - Panel de Pruebas")
        self.root.geometry("700x650")
        self.root.minsize(600, 500)
        
        # Color palette
        self.bg_color = "#f4f6f9"
        self.primary_color = "#3b82f6" # Azul moderno
        self.text_color = "#1f2937"
        self.btn_active_bg = "#2563eb"
        
        self.root.configure(bg=self.bg_color)
        
        # --- Cabecera ---
        header_frame = tk.Frame(root, bg="#1e293b", height=60)
        header_frame.pack(fill="x", side="top")
        
        header_label = tk.Label(
            header_frame, 
            text="YOLO11 Segmentation - Evaluador de Modelos", 
            fg="white", 
            bg="#1e293b", 
            font=("Helvetica", 14, "bold")
        )
        header_label.pack(pady=15)
        
        # --- Cuerpo / Contenedor principal ---
        main_frame = tk.Frame(root, bg=self.bg_color, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # 1. Configuración de pesos (.pt)
        weights_frame = tk.LabelFrame(main_frame, text=" 1. Pesos del Modelo (.pt) ", font=("Helvetica", 10, "bold"), bg=self.bg_color, fg=self.text_color, padx=10, pady=10)
        weights_frame.pack(fill="x", pady=5)
        
        self.weights_var = tk.StringVar(value="modelofinal.pt")
        self.weights_entry = tk.Entry(weights_frame, textvariable=self.weights_var, font=("Helvetica", 10), width=50)
        self.weights_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        weights_btn = tk.Button(weights_frame, text="Examinar...", command=self.browse_weights, bg="#e2e8f0", activebackground="#cbd5e1", font=("Helvetica", 9))
        weights_btn.pack(side="right")
        
        # 2. Configuración de dataset (data.yaml)
        data_frame = tk.LabelFrame(main_frame, text=" 2. Configuración del Dataset (data.yaml) - Opcional ", font=("Helvetica", 10, "bold"), bg=self.bg_color, fg=self.text_color, padx=10, pady=10)
        data_frame.pack(fill="x", pady=5)
        
        self.data_var = tk.StringVar()
        # Intentar autodetectar data.yaml local
        self.autodetect_yaml()
        
        self.data_entry = tk.Entry(data_frame, textvariable=self.data_var, font=("Helvetica", 10), width=50)
        self.data_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        data_btn = tk.Button(data_frame, text="Examinar...", command=self.browse_data, bg="#e2e8f0", activebackground="#cbd5e1", font=("Helvetica", 9))
        data_btn.pack(side="right")
        
        # 3. Configuración de imagen de prueba
        image_frame = tk.LabelFrame(main_frame, text=" 3. Imagen o Carpeta para Inferencia Visual ", font=("Helvetica", 10, "bold"), bg=self.bg_color, fg=self.text_color, padx=10, pady=10)
        image_frame.pack(fill="x", pady=5)
        
        self.image_var = tk.StringVar()
        self.image_entry = tk.Entry(image_frame, textvariable=self.image_var, font=("Helvetica", 10), width=50)
        self.image_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        image_btn = tk.Button(image_frame, text="Examinar...", command=self.browse_image, bg="#e2e8f0", activebackground="#cbd5e1", font=("Helvetica", 9))
        image_btn.pack(side="right")
        
        # --- Botones de Acción ---
        actions_frame = tk.Frame(main_frame, bg=self.bg_color, pady=10)
        actions_frame.pack(fill="x")
        
        self.val_btn = tk.Button(
            actions_frame, 
            text="Evaluar Métricas (mAP en Test)", 
            command=self.run_validation_thread, 
            bg=self.primary_color, 
            fg="white", 
            activebackground=self.btn_active_bg,
            activeforeground="white",
            font=("Helvetica", 10, "bold"),
            padx=10,
            pady=5
        )
        self.val_btn.pack(side="left", padx=(0, 10), fill="x", expand=True)
        
        self.predict_btn = tk.Button(
            actions_frame, 
            text="Probar Inferencia (Predict Visual)", 
            command=self.run_prediction_thread, 
            bg="#10b981", # Verde
            fg="white", 
            activebackground="#059669",
            activeforeground="white",
            font=("Helvetica", 10, "bold"),
            padx=10,
            pady=5
        )
        self.predict_btn.pack(side="right", padx=(10, 0), fill="x", expand=True)
        
        # --- Consola de Salida / Logs ---
        log_frame = tk.LabelFrame(main_frame, text=" Consola de Salida ", font=("Helvetica", 10, "bold"), bg=self.bg_color, fg=self.text_color, padx=5, pady=5)
        log_frame.pack(fill="both", expand=True, pady=10)
        
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap="word", bg="#0f172a", fg="#f8fafc", font=("Consolas", 9), insertbackground="white")
        self.log_area.pack(fill="both", expand=True)
        
        # Redirigir stdout al área de texto de la GUI
        sys.stdout = StdoutRedirector(self.log_area)
        sys.stderr = StdoutRedirector(self.log_area)
        
        print("Sistema iniciado. Selecciona tus archivos y presiona un botón para comenzar.")

    def autodetect_yaml(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Intentar buscar una carpeta de dataset local
        possible_dirs = [
            "food.v1i.yolo-segmentation",
            "Food types.v3i.yolo-segmentation"
        ]
        for p_dir in possible_dirs:
            yaml_path = os.path.join(base_dir, p_dir, "data.yaml")
            if os.path.exists(yaml_path):
                self.data_var.set(yaml_path)
                break

    def browse_weights(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar Pesos (.pt)",
            filetypes=[("Modelos YOLO (*.pt)", "*.pt")]
        )
        if file_path:
            self.weights_var.set(file_path)
            
    def browse_data(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar data.yaml",
            filetypes=[("Configuración YAML (*.yaml)", "*.yaml")]
        )
        if file_path:
            self.data_var.set(file_path)
            
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen de Prueba",
            filetypes=[
                ("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )
        if file_path:
            self.image_var.set(file_path)

    # Métodos para correr en hilos secundarios y no congelar la GUI
    def run_validation_thread(self):
        self.set_buttons_state("disabled")
        t = threading.Thread(target=self.run_validation)
        t.daemon = True
        t.start()

    def run_prediction_thread(self):
        self.set_buttons_state("disabled")
        t = threading.Thread(target=self.run_prediction)
        t.daemon = True
        t.start()

    def set_buttons_state(self, state):
        self.val_btn.configure(state=state)
        self.predict_btn.configure(state=state)

    def run_validation(self):
        weights = self.weights_var.get()
        data = self.data_var.get()
        
        print("\n\n==========================================")
        print("=== EVALUACIÓN DE MÉRICAS (TEST SPLIT) ===")
        print("==========================================")
        
        if not weights or not os.path.exists(weights):
            print(f"Error: No se encontró el archivo de pesos '{weights}'.")
            self.set_buttons_state("normal")
            return
            
        if not data or not os.path.exists(data):
            print(f"Error: Debes proporcionar un archivo 'data.yaml' válido.")
            self.set_buttons_state("normal")
            return
            
        try:
            print(f"Cargando modelo: {weights}...")
            model = YOLO(weights)
            print("Iniciando validación...")
            metrics = model.val(data=data, split='test', device='cpu')
            
            print("\n============ RESULTADOS DE EVALUACIÓN ============")
            print(f"  Box mAP50:      {metrics.box.map50:.4f}")
            print(f"  Box mAP50-95:   {metrics.box.map:.4f}")
            print(f"  Mask mAP50:     {metrics.mask.map50:.4f}")
            print(f"  Mask mAP50-95:  {metrics.mask.map:.4f}")
            print("==================================================")
            messagebox.showinfo("Evaluación Completa", "Las métricas han sido calculadas con éxito.")
        except Exception as e:
            print(f"Error durante el cálculo de métricas: {e}")
            messagebox.showerror("Error", f"Ocurrió un error: {e}")
            
        self.set_buttons_state("normal")

    def run_prediction(self):
        weights = self.weights_var.get()
        image = self.image_var.get()
        
        print("\n\n=========================================")
        print("=== INFERENCIA VISUAL (PREDICT IMAGE) ===")
        print("=========================================")
        
        if not weights or not os.path.exists(weights):
            print(f"Error: No se encontró el archivo de pesos '{weights}'.")
            self.set_buttons_state("normal")
            return
            
        if not image or not os.path.exists(image):
            print(f"Error: Debes proporcionar una imagen de prueba válida.")
            self.set_buttons_state("normal")
            return
            
        try:
            print(f"Cargando modelo: {weights}...")
            model = YOLO(weights)
            print(f"Ejecutando inferencia en: {image}...")
            
            results = model.predict(
                source=image,
                save=True,
                project='test_results',
                name='predictions',
                exist_ok=True,
                device='cpu'
            )
            
            print("\nDetecciones completadas:")
            for idx, result in enumerate(results):
                if len(result.boxes) == 0:
                    print(f"  No se detectaron elementos en la imagen.")
                else:
                    detections = []
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        detections.append(f"{cls_name} ({conf:.2f})")
                    print(f"  Imagen: {os.path.basename(result.path)}")
                    print(f"  Detecciones: {', '.join(detections)}")
            
            # Avisar dónde se guardaron
            save_dir = os.path.join("test_results", "predictions")
            print(f"\nImagen procesada guardada en: {os.path.abspath(save_dir)}")
            messagebox.showinfo("Inferencia Completa", f"Predicción exitosa.\nLos resultados se guardaron en:\n{save_dir}")
        except Exception as e:
            print(f"Error durante la inferencia: {e}")
            messagebox.showerror("Error", f"Ocurrió un error: {e}")
            
        self.set_buttons_state("normal")

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        
    def write(self, string):
        self.text_widget.insert('end', string)
        self.text_widget.see('end')
        
    def flush(self):
        pass

def test_model_cli(weights_path, data_yaml_path, test_source_path=None):
    # Función CLI original para fallback en consola si hay argumentos
    print(f"Cargando modelo YOLO desde: {weights_path}...")
    if not os.path.exists(weights_path):
        print(f"Error: No se encontró el archivo de pesos en '{weights_path}'")
        return
        
    model = YOLO(weights_path)
    
    if data_yaml_path and os.path.exists(data_yaml_path):
        print(f"\n=== Iniciando Evaluación en el Split de Prueba ('test') ===")
        metrics = model.val(data=data_yaml_path, split='test', device='cpu')
        print("\n================ METRICAS DE PRUEBA (TEST SPLIT) ================")
        print(f"  Box mAP50:      {metrics.box.map50:.4f}")
        print(f"  Box mAP50-95:   {metrics.box.map:.4f}")
        print(f"  Mask mAP50:     {metrics.mask.map50:.4f}")
        print(f"  Mask mAP50-95:  {metrics.mask.map:.4f}")
        print("================================================================")
        
    if test_source_path:
        print(f"\n=== Inferencia en: {test_source_path} ===")
        results = model.predict(source=test_source_path, save=True, project='test_results', name='predictions', exist_ok=True, device='cpu')
        for idx, result in enumerate(results):
            if len(result.boxes) == 0:
                print(f"  No se detectaron objetos.")
            else:
                detections = []
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    detections.append(f"{cls_name} ({conf:.2f})")
                print(f"  Detectado: {', '.join(detections)}")

if __name__ == "__main__":
    # Si se pasan argumentos por línea de comandos, usar el modo consola CLI
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Script CLI de Pruebas YOLO11 Segment")
        parser.add_argument("--weights", type=str, default="modelofinal.pt")
        parser.add_argument("--data", type=str)
        parser.add_argument("--image", type=str)
        args = parser.parse_args()
        test_model_cli(args.weights, args.data, args.image)
    else:
        # Por defecto, iniciar la interfaz gráfica de usuario si tkinter está disponible
        if GUI_AVAILABLE:
            root = tk.Tk()
            app = GUIApp(root)
            root.mainloop()
        else:
            print("Error: Interfaz gráfica (Tkinter) no disponible. Ejecuta el script con argumentos CLI:")
            print("Ejemplo: python test_model.py --weights modelofinal.pt --image mi_imagen.jpg")
