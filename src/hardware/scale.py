import serial
import logging
import random
import sys
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DigitalScale")

class DigitalScale:
    def __init__(self, port=settings.SERIAL_PORT, baudrate=settings.BAUD_RATE, timeout=settings.TIMEOUT):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.mock_mode = settings.MOCK_MODE

    def connect(self):
        """Conecta con la balanza digital serial o activa modo mock si corresponde."""
        if self.mock_mode:
            logger.info("Modo simulación (MOCK_MODE) activo. Balanza virtual iniciada.")
            return True

        try:
            logger.info(f"Intentando conectar a la balanza en puerto {self.port} ({self.baudrate} baud)...")
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            logger.info("Balanza física conectada exitosamente.")
            return True
        except serial.SerialException as e:
            logger.warning(f"No se pudo conectar a la balanza física en {self.port}. Detalle: {e}")
            logger.info("Cambiando automáticamente a modo simulación (MockScale).")
            self.mock_mode = True
            return True

    def read_weight(self, preset_weight=None):
        """
        Lee el peso de la balanza.
        Si está en modo mock, usa el preset_weight si está provisto, pide el peso por consola o genera un valor de prueba realista.
        """
        if preset_weight is not None:
            try:
                weight = float(preset_weight)
                logger.info(f"Balanza: usando peso predefinido desde la GUI: {weight} g.")
                return weight
            except ValueError:
                logger.warning(f"El peso predefinido '{preset_weight}' no es un número válido. Usando lectura normal.")

        if self.mock_mode:
            return self._read_mock_weight()

        if self.ser is not None and self.ser.is_open:
            try:
                # La lectura del peso depende del protocolo de la balanza.
                # Muchas envían strings terminados en '\r\n' continuamente (ej. "  0.345 kg\r\n").
                # Otras requieren enviar un byte de comando para solicitar el dato (ej. b'W\r').
                
                # Consumir datos del buffer
                self.ser.reset_input_buffer()
                
                # Enviar solicitud si la balanza requiere trigger (ejemplo común)
                # self.ser.write(b'W\r') 
                
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    logger.info(f"Datos recibidos crudos de balanza: '{line}'")
                    # Extraer el valor numérico en gramos
                    # Ejemplo de conversión rápida para "  324.5 g" o "0.324 kg"
                    weight = self._parse_weight_string(line)
                    if weight is not None:
                        return weight
                
                logger.warning("Balanza física retornó datos vacíos o ilegibles. Reintentando...")
            except Exception as e:
                logger.error(f"Error al leer de la balanza serial física: {e}")
            
            logger.warning("Fallo en lectura física. Usando peso de simulación como respaldo.")
            return self._read_mock_weight(interactive=False)

        logger.warning("El puerto serial de la balanza no está abierto. Usando simulación.")
        return self._read_mock_weight()

    def _parse_weight_string(self, text):
        """Parsea strings comunes de balanza serial y convierte a gramos."""
        try:
            # Eliminar caracteres no numéricos excepto puntos
            # Si el string contiene 'kg', el peso leído debe multiplicarse por 1000.
            is_kg = 'kg' in text.lower()
            
            # Limpieza básica del texto
            cleaned = "".join([c for c in text if c.isdigit() or c in ".-"])
            val = float(cleaned)
            
            if is_kg:
                val = val * 1000.0
            
            return val
        except ValueError:
            return None

    def _read_mock_weight(self, interactive=True):
        """Pide peso interactivo si es posible o genera uno aleatorio/por defecto."""
        # Evitar bloquear en scripts de test no interactivos
        if interactive and sys.stdin.isatty():
            try:
                print("\n[SIMULADOR BALANZA]")
                entrada = input("Ingrese el peso de las sobras (en gramos) [Default: 350g]: ").strip()
                if not entrada:
                    weight = 350.0
                else:
                    weight = float(entrada)
                logger.info(f"Balanza Simulación: {weight} g registrados.")
                return weight
            except (ValueError, KeyboardInterrupt):
                weight = 350.0
                logger.info(f"Entrada inválida o cancelada. Usando peso default: {weight} g.")
                return weight
        else:
            # Si no es interactivo, generar un peso entre 150g y 500g
            weight = round(random.uniform(150.0, 500.0), 1)
            logger.info(f"Balanza Simulación (Automático): {weight} g registrados.")
            return weight

    def close(self):
        """Cierra el puerto serial."""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            self.ser = None
            logger.info("Conexión serial con balanza física cerrada.")
