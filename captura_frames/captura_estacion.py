import cv2
import time
import os
import serial 

def main():
    carpeta_dataset = "dataset_charolas"
    if not os.path.exists(carpeta_dataset):
        os.makedirs(carpeta_dataset)

    try:
        # CAMBIA 'COM3' POR EL PUERTO DE ARDUINO
        arduino = serial.Serial('COM15', 9600, timeout=1)
        time.sleep(2)
        print("Arduino conectado exitosamente.")
    except Exception as e:
        print(f"Error conectando al Arduino: {e}")
        return

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    roi_x, roi_y = 20, 40
    roi_w, roi_h = 540, 380
    
    UMBRAL_PIXELES = 6000 
    TIEMPO_ESTABILIZACION = 1.5 
    TIEMPO_ENFRIAMIENTO = 7.0 

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    estado_sistema = "ESPERANDO"
    tiempo_deteccion = 0
    tiempo_salida = 0
    contador_muestras = 0 

    print("Estación automatizada iniciada. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        mascara = fgbg.apply(roi)
        _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_total = 0
        for c in contornos:
            area = cv2.contourArea(c)
            if area > 500:
                area_total += area

        # LÓGICA DE LA ESTACIÓN DE ESTADOS
        if estado_sistema == "ESPERANDO" and area_total > UMBRAL_PIXELES:
            estado_sistema = "ESTABILIZANDO"
            tiempo_deteccion = time.time()
            print("¡Objeto detectado! Esperando estabilización...")

        elif estado_sistema == "ESTABILIZANDO":
            if area_total < UMBRAL_PIXELES and (time.time() - tiempo_deteccion) < 0.5:
                estado_sistema = "ESPERANDO"
            elif time.time() - tiempo_deteccion >= TIEMPO_ESTABILIZACION:
                estado_sistema = "CAPTURANDO"

        elif estado_sistema == "CAPTURANDO":
            print("Capturando imágenes...")
            contador_muestras += 1
            subcarpeta = os.path.join(carpeta_dataset, f"charola_{contador_muestras}_{int(time.time())}")
            os.makedirs(subcarpeta)

            for i in range(5):
                ret, f_captura = cap.read()
                if ret:
                    foto_roi = f_captura[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    nombre_archivo = os.path.join(subcarpeta, f"imagen_{i+1}.jpg")
                    cv2.imwrite(nombre_archivo, foto_roi)
                    time.sleep(0.1)
            
            # ACTIVACIÓN DEL BUZZER FÍSICO CON ARDUINO
            print("¡CAPTURA COMPLETA! Enviando señal al Arduino...")
            arduino.write(b'1') # Enviamos el caracter '1' convertido a bytes

            estado_sistema = "COMPLETO"

        elif estado_sistema == "COMPLETO":
            if area_total < 2000: 
                estado_sistema = "ENFRIAMIENTO"
                tiempo_salida = time.time()

        elif estado_sistema == "ENFRIAMIENTO":
            if time.time() - tiempo_salida >= TIEMPO_ENFRIAMIENTO:
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
                estado_sistema = "ESPERANDO"
                print("Estación despejada. Lista para la siguiente charola.\n")

        # DIBUJOS EN EN PANTALLA
        if estado_sistema == "ESPERANDO": color = (0, 255, 0)
        elif estado_sistema == "ESTABILIZANDO": color = (0, 255, 255)
        elif estado_sistema == "CAPTURANDO": color = (0, 0, 255)
        elif estado_sistema == "COMPLETO": color = (255, 0, 0)
        elif estado_sistema == "ENFRIAMIENTO": color = (255, 165, 0)

        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), color, 2)
        cv2.putText(frame, f"ESTADO: {estado_sistema}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if estado_sistema != "ENFRIAMIENTO":
            cv2.putText(frame, f"Area ROI: {int(area_total)} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            faltan = round(TIEMPO_ENFRIAMIENTO - (time.time() - tiempo_salida), 1)
            cv2.putText(frame, f"Bloqueo: {faltan}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Estacion de Captura UNCP - Automatizada', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Limpieza final
    cap.release()
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()