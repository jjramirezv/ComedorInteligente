import cv2
import logging
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import customtkinter as ctk 
# Asegurar importación de módulos locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from src.hardware.camera import USBCamera
from src.hardware.scale import DigitalScale
from src.segmentation.food_segmenter import FoodSegmenter
from src.estimation.weight_estimator import WeightEstimator
from src.utils.image_processing import draw_annotations, save_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("UNCP_GUI_Main")

ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue")

class UNCPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UNCP - Visión Artificial para Estimación de Desperdicios")
        self.geometry("1200x750") 
        self.configure(bg="#121212")
        
        self.minsize(1100, 700)
        
        self.camera = USBCamera()
        self.scale = DigitalScale()
        self.segmenter = FoodSegmenter()
        self.estimator = WeightEstimator()
        
        self.camera_active = False
        self.current_frame = None       # Frame actual en crudo
        self.displayed_image = None     # Imagen mostrada en panel (original o procesada)
        self.live_feed_running = True   # Si el stream de cámara en vivo está corriendo
        self.scale_weight_var = tk.StringVar(value="350.0") # Peso por defecto
        
        self.init_hardware()
        
        self.create_widgets()
        
        # Iniciar loop de video
        if self.camera_active:
            self.update_video_feed()
        else:
            self.show_static_simulated_frame()

    def init_hardware(self):
        """Inicializa la cámara, balanza y carga el modelo YOLO11."""
        self.camera_active = self.camera.initialize()
        if not self.camera_active:
            logger.warning("Cámara física no disponible. Cambiando a modo simulación.")
            
        self.scale.connect()
        
        # Cargar modelo YOLO11 (modelofinal.pt)
        try:
            self.segmenter.load_model()
        except Exception as e:
            logger.error(f"Error crítico al cargar el modelo: {e}")
            messagebox.showerror("Error de Inicialización", f"No se pudo cargar el modelo YOLO11.\n\nDetalle: {e}")
            self.destroy()
            sys.exit(1)

    def create_widgets(self):
        # --- Layout Principal ---
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=3) # Columna de video (más ancha)
        self.grid_columnconfigure(1, weight=1) # Columna de controles
        
        # Header
        self.title_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.title_frame.grid(row=0, column=0, columnspan=2, pady=15, sticky="ew")
        self.title_lbl = ctk.CTkLabel(self.title_frame, text="COMEDOR UNCP - SISTEMA DE ESTIMACIÓN DE DESPERDICIOS", 
                                      font=ctk.CTkFont(size=20, weight="bold"))
        self.title_lbl.pack()

        # Columna Izquierda: Visor de Imagen/Video
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        self.visor_title = ctk.CTkLabel(self.left_frame, text="Feed de Cámara en Vivo", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00e5ff")
        self.visor_title.pack(pady=(15, 5))
        
        self.video_label = ctk.CTkLabel(self.left_frame, text="") # Contenedor del video
        self.video_label.pack(fill="both", expand=True, padx=15, pady=15)

        # Botón de regreso (oculto por defecto)
        self.btn_live_feed = ctk.CTkButton(self.left_frame, text="Regresar a Feed en Vivo", command=self.resume_live_feed)

        # Columna Derecha: Controles
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=1, column=1, padx=(0, 20), pady=10, sticky="nsew")
        
        # Sección 1: Balanza
        self.scale_lbl = ctk.CTkLabel(self.right_frame, text="Entrada de Balanza Digital (g)", font=ctk.CTkFont(weight="bold"))
        self.scale_lbl.pack(pady=(20, 5), padx=20, anchor="w")
        
        self.scale_entry = ctk.CTkEntry(self.right_frame, textvariable=self.scale_weight_var, 
                                        font=ctk.CTkFont(size=18), justify="center", height=40)
        self.scale_entry.pack(pady=5, padx=20, fill="x")

        # Sección 2: Operaciones
        self.btn_estimate = ctk.CTkButton(self.right_frame, text="⚡ CAPTURAR Y ESTIMAR", 
                                          fg_color="#107c41", hover_color="#0b5a30", height=45,
                                          command=self.process_capture)
        self.btn_estimate.pack(pady=(30, 10), padx=20, fill="x")

        self.btn_upload = ctk.CTkButton(self.right_frame, text="📁 ADJUNTAR IMAGEN LOCAL", 
                                        height=40, command=self.upload_local_image)
        self.btn_upload.pack(pady=10, padx=20, fill="x")

        # Sección 3: Resultados (Treeview clásico estilizado)
        self.results_lbl = ctk.CTkLabel(self.right_frame, text="Distribución de Residuos", font=ctk.CTkFont(weight="bold"))
        self.results_lbl.pack(pady=(30, 10), padx=20, anchor="w")
        
        # CustomTkinter no tiene Treeview, usamos el de ttk adaptando los colores
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", borderwidth=0)
        style.configure("Treeview.Heading", background="#1f538d", foreground="white", relief="flat")
        style.map("Treeview", background=[("selected", "#1f538d")])

        columns = ("categoria", "peso", "porcentaje")
        self.results_table = ttk.Treeview(self.right_frame, columns=columns, show="headings", height=8)
        self.results_table.heading("categoria", text="Categoría")
        self.results_table.heading("peso", text="Peso (g)")
        self.results_table.heading("porcentaje", text="%")
        
        self.results_table.column("categoria", width=120)
        self.results_table.column("peso", width=80, anchor="center")
        self.results_table.column("porcentaje", width=60, anchor="center")
        self.results_table.pack(pady=5, padx=20, fill="both", expand=True)

        # Footer
        self.status_lbl = ctk.CTkLabel(self, text="Inicializando sistema...", text_color="gray")
        self.status_lbl.grid(row=2, column=0, columnspan=2, sticky="w", padx=20, pady=5)

    def update_video_feed(self):
        """Lee el frame de la cámara web, lo convierte a formato tkinter y se agenda de nuevo."""
        if self.live_feed_running and self.camera_active:
            ret, frame = self.camera.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.display_opencv_frame(frame)
            
            # Agendar lectura en 33ms (~30 FPS)
            self.after(33, self.update_video_feed)

    def show_static_simulated_frame(self):
        """Genera y muestra una bandeja simulada cuando no hay hardware físico de cámara."""
        self.current_frame = self.camera.capture_frame()
        self.display_opencv_frame(self.current_frame)
        self.status_lbl.configure(text="Listo. Operando en modo Simulación de Bandeja Cenital.")

    def display_opencv_frame(self, frame):
        """Procesa y despliega una imagen OpenCV (BGR) en el Label de la interfaz."""
        # Redimensionar dinámicamente para que quepa en el panel manteniendo proporción
        panel_w = self.video_label.winfo_width()
        panel_h = self.video_label.winfo_height()
        
        # Ajustes por defecto si el widget aún no está dibujado
        if panel_w <= 1 or panel_h <= 1:
            panel_w, panel_h = 750, 420
            
        img_h, img_w = frame.shape[:2]
        
        # Calcular proporciones de escalado
        ratio_w = panel_w / img_w
        ratio_h = panel_h / img_h
        ratio = min(ratio_w, ratio_h)
        
        new_w = max(int(img_w * ratio) - 10, 100)
        new_h = max(int(img_h * ratio) - 10, 100)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convertir BGR a RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convertir a imagen Tkinter
        pil_img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=pil_img)
        self.video_label.configure(image=self.photo)
        self.displayed_image = frame # Guardar referencia de imagen mostrada

    def process_capture(self):
        """Captura el frame actual y ejecuta todo el análisis y visualización."""
        if self.current_frame is None:
            messagebox.showerror("Error", "No hay imagen disponible para estimar.")
            return

        # Pausar feed de video en vivo
        self.live_feed_running = False
        
        # Leer el peso de la balanza ingresado en el campo de la interfaz
        weight_input = self.scale_weight_var.get().strip()
        try:
            total_weight = float(weight_input)
        except ValueError:
            messagebox.showerror("Error de Peso", "Por favor ingrese un valor numérico válido para el peso (en gramos).")
            self.live_feed_running = True
            if self.camera_active:
                self.update_video_feed()
            return

        self.status_lbl.configure(text="Ejecutando segmentación YOLO11...")
        self.update_idletasks() # Forzar dibujado del texto de carga

        # 1. Ejecutar lectura con balanza (inyectando el peso de la interfaz)
        scale_weight = self.scale.read_weight(preset_weight=total_weight)

        # 2. Ejecutar segmentación de alimentos
        detections, annotated_raw = self.segmenter.segment(self.current_frame)

        # 3. Estimar pesos por densidad
        self.status_lbl.configure(text="Distribuyendo gramaje por densidad...")
        estimation_results = self.estimator.estimate_weights(detections, scale_weight)

        # 4. Generar imagen final con HUD visual premium
        annotated_hud = draw_annotations(self.current_frame, detections, estimation_results)

        # 5. Desplegar en la interfaz la imagen final procesada
        self.display_opencv_frame(annotated_hud)
        self.visor_title.configure(text="Resultado del Análisis (Segmentación + Peso)")
        
        # Habilitar botón para regresar a feed en vivo si hay cámara real
        if self.camera_active:
            self.btn_live_feed.pack(side=tk.RIGHT, padx=15)

        # 6. Actualizar la tabla de resultados
        self.update_results_table(estimation_results)

        # 7. Guardar la imagen en el historial captures/
        path_raw = save_frame(self.current_frame, prefix="captura_original")
        path_annotated = save_frame(annotated_hud, prefix="captura_estimada")
        
        logger.info(f"Historial guardado: {path_raw} | {path_annotated}")
        self.status_lbl.configure(text=f"Análisis completado. Imágenes guardadas en captures/.")

    def upload_local_image(self):
        """Abre un explorador de archivos para subir una foto local a analizar."""
        file_path = filedialog.askopenfilename(
            title="Seleccionar bandeja de retorno",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        # Cargar imagen usando OpenCV
        uploaded_img = cv2.imread(file_path)
        if uploaded_img is None:
            messagebox.showerror("Error de Archivo", "No se pudo cargar la imagen seleccionada.")
            return

        # Pausar feed de video
        self.live_feed_running = False
        self.current_frame = uploaded_img
        
        # Mostrar la imagen cargada en bruto en la pantalla
        self.display_opencv_frame(self.current_frame)
        self.visor_title.configure(text=f"Imagen Cargada: {os.path.basename(file_path)}")
        self.status_lbl.configure(text="Imagen local cargada. Presione '⚡ CAPTURAR Y ESTIMAR' para procesarla.")
        
        # Habilitar botón para regresar a feed en vivo
        if self.camera_active:
            self.btn_live_feed.pack(side=tk.RIGHT, padx=15)

    def resume_live_feed(self):
        """Reanuda el video continuo en vivo de la cámara."""
        self.live_feed_running = True
        self.btn_live_feed.pack_forget()
        self.visor_title.configure(text="Feed de Cámara en Vivo")
        self.status_lbl.configure(text="Listo. Visualizando feed en tiempo real.")
        
        # Limpiar tabla de resultados
        for item in self.results_table.get_children():
            self.results_table.delete(item)
            
        self.update_video_feed()

    def update_results_table(self, results):
        """Actualiza el Treeview con los valores estimados."""
        # Limpiar
        for item in self.results_table.get_children():
            self.results_table.delete(item)
            
        # Rellenar con nuevos resultados
        for class_name, data in results.items():
            category_clean = class_name.replace("_", " ").capitalize()
            weight_g = f"{data['estimated_weight_g']:.1f} g"
            percent = f"{data['percentage_weight']:.1f} %"
            
            self.results_table.insert("", tk.END, values=(category_clean, weight_g, percent))

    def destroy(self):
        """Detiene recursos en el cierre de la ventana."""
        self.live_feed_running = False
        self.camera.release()
        self.scale.close()
        super().destroy()

if __name__ == "__main__":
    app = UNCPApp()
    app.mainloop()
