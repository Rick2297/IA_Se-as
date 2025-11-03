import tkinter as tk
import customtkinter as ctk
from PIL import Image
import cv2
import threading
import mediapipe as mp
import numpy as np
import os
import time
from queue import Queue, Empty, Full
from tensorflow.keras.models import load_model
from gtts import gTTS
import playsound

# --- 1. CONFIGURACIÓN Y PARÁMETROS ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
CARPETA_MODELO = 'modelo_entrenado'
RUTA_MODELO = os.path.join(CARPETA_MODELO, 'modelo_gestos.keras')
RUTA_ETIQUETAS = os.path.join(CARPETA_MODELO, 'etiquetas.npy')
CARPETA_AUDIOS = 'audios_mp3'

# Parámetros de MediaPipe
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# --- PARÁMETROS DE PREDICCIÓN MEJORADOS ---
SEQUENCE_LENGTH = 30
PREDICTION_THRESHOLD = 0.85
PREDICTION_COOLDOWN = 1.5   # Tiempo de espera entre señas.
SIGN_DISPLAY_DURATION = 2.0  # Duración del texto en pantalla.

# --- 2. FUNCIONES AUXILIARES Y WORKERS DE AUDIO ---

if not os.path.exists(CARPETA_AUDIOS):
    os.makedirs(CARPETA_AUDIOS)

# Cola 1: Archivos MP3 listos para reproducir
audio_queue = Queue()

def audio_worker():
    """Hilo trabajador que reproduce archivos de audio desde la audio_queue."""
    while True:
        filename = audio_queue.get()
        if filename is None: break
        try:
            playsound.playsound(filename, True)
            if os.path.exists(filename): os.remove(filename)
        except Exception as e:
            print(f"[ERROR playsound] {e}")
        finally:
            audio_queue.task_done()

def generate_audio_file(text):
    """Genera el MP3 y lo pone en la cola de reproducción (audio_queue)"""
    try:
        filename = os.path.join(CARPETA_AUDIOS, f"audio_{int(time.time() * 1000)}.mp3")
        tts = gTTS(text=text, lang='es', slow=False)
        tts.save(filename)
        audio_queue.put(filename) # Pone el filename en la cola de PLAYBACK
    except Exception as e:
        print(f"[ERROR gTTS] {e}")

# --- Funciones de MediaPipe ---
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def draw_styled_landmarks(image, results):
    drawing_spec_points = mp.solutions.drawing_utils.DrawingSpec(color=(100,100,100), thickness=1, circle_radius=2)
    drawing_spec_lines = mp.solutions.drawing_utils.DrawingSpec(color=(150,0,150), thickness=1, circle_radius=1)
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, drawing_spec_points, drawing_spec_lines)
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, drawing_spec_points, drawing_spec_lines)

# --- 3. CLASE PRINCIPAL ---
class NeuroLESSAApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NeuroLESSA - Intérprete de Señas IA")
        self.geometry("1100x720")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Variables de estado
        self.is_camera_on = False
        self.cap = None
        self.update_id = None
        
        # Hilos
        self.processing_thread = None
        self.prediction_thread = None
        
        # Lógica de señas (estado compartido)
        self.sequence = []
        self.current_sign_text = ""
        self.last_prediction_time = 0.0
        self.sign_display_start_time = 0.0
        
        # Buffers y Colas (Queues)
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.keypoint_queue = Queue(maxsize=1) # Para pasar keypoints a la IA (maxsize=1 para real-time)
        self.gtts_queue = Queue()              # Para pasar texto a gTTS

        # Carga del modelo
        try:
            self.modelo = load_model(RUTA_MODELO)
            self.actions = np.load(RUTA_ETIQUETAS)
            print("[INFO] Modelo y etiquetas cargados.")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo: {e}")
            self.modelo, self.actions = None, []
            
        # Inicialización de MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        # --- Iniciar hilos de audio (workers) ---
        threading.Thread(target=audio_worker, daemon=True).start()
        threading.Thread(target=self.gtts_worker_method, daemon=True).start()
        
        # Crear la interfaz
        self.create_widgets()
        self.video_width, self.video_height = 1, 1

    def create_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # Frame izquierdo
        self.frame_izquierdo = ctk.CTkFrame(self, width=250, corner_radius=20)
        self.frame_izquierdo.grid(row=0, column=0, padx=20, pady=20, sticky="ns")
        self.frame_izquierdo.grid_propagate(False)
        self.label_titulo = ctk.CTkLabel(self.frame_izquierdo, text="🤖 NeuroLESSA", font=ctk.CTkFont(size=28, weight="bold"))
        self.label_titulo.pack(pady=(20,10), padx=10)
        self.label_subtitulo = ctk.CTkLabel(self.frame_izquierdo, text="Intérprete de Lengua de Señas", font=ctk.CTkFont(size=14))
        self.label_subtitulo.pack(pady=(0,30), padx=10)
        self.btn_toggle_camera = ctk.CTkButton(self.frame_izquierdo, text="Iniciar Cámara", command=self.toggle_camera, height=40)
        self.btn_toggle_camera.pack(pady=20, padx=20, fill="x")
        self.status_frame = ctk.CTkFrame(self.frame_izquierdo)
        self.status_frame.pack(pady=20, padx=20, fill="x")
        self.label_status_title = ctk.CTkLabel(self.status_frame, text="ÚLTIMA SEÑA DETECTADA:", font=ctk.CTkFont(size=12, weight="bold"))
        self.label_status_title.pack(pady=(10,5))
        self.label_detected_sign = ctk.CTkLabel(self.status_frame, text="---", font=ctk.CTkFont(size=36, weight="bold"), text_color="#3498db", width=200, wraplength=200, anchor="center")
        self.label_detected_sign.pack(pady=(5,15))
        self.btn_salir = ctk.CTkButton(self.frame_izquierdo, text="Salir", command=self.on_closing, fg_color="#D32F2F", hover_color="#B71C1C", height=40)
        self.btn_salir.pack(side="bottom", pady=20, padx=20, fill="x")
        # Frame derecho
        self.frame_derecho = ctk.CTkFrame(self, corner_radius=20)
        self.frame_derecho.grid(row=0, column=1, padx=(0,20), pady=20, sticky="nsew")
        self.label_video = ctk.CTkLabel(self.frame_derecho, text="Cámara Apagada", font=ctk.CTkFont(size=16))
        self.label_video.pack(expand=True, fill="both", padx=10, pady=10)
        self.label_video.bind("<Configure>", self.on_video_resize)

    def on_video_resize(self, event):
        self.video_width, self.video_height = event.width, event.height

    def toggle_camera(self):
        if self.is_camera_on: 
            self.stop_camera()
        else: 
            self.start_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara")
            self.cap = None
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_camera_on = True
        self.sequence.clear()
        
        # Reseteamos los estados al iniciar la cámara
        self.last_prediction_time = 0.0
        self.current_sign_text = ""
        self.label_detected_sign.configure(text="---")

        # Iniciar hilos de procesamiento
        self.processing_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
        self.prediction_thread = threading.Thread(target=self.prediction_loop, daemon=True)
        
        self.processing_thread.start()
        self.prediction_thread.start()
        
        self.btn_toggle_camera.configure(text="Detener Cámara")
        self.update_gui_loop() # Iniciar el bucle de la GUI

    def stop_camera(self):
        """Detiene la cámara y los hilos de procesamiento de video/predicción."""
        print("[INFO] Deteniendo la cámara...")
        self.is_camera_on = False
        
        # Detener el bucle de la GUI
        if self.update_id:
            self.after_cancel(self.update_id)
            self.update_id = None
            
        # Enviar señal de parada al hilo de predicción
        try:
            self.keypoint_queue.put_nowait(None) 
        except Full:
            pass # No importa si está llena, la bandera is_camera_on lo detendrá
        
        # --- CAMBIO CLAVE: Liberar la cámara PRIMERO ---
        # Esto desbloqueará el self.cap.read() en video_processing_loop
        if self.cap:
            self.cap.release()
            
        # Esperar a que los hilos terminen (ahora deberían terminar rápido)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0) 
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=1.0)
            
        self.cap = None
        self.processing_thread = None
        self.prediction_thread = None
            
        # Mostrar pantalla negra (seguro, estamos en el hilo principal)
        frame_black = Image.new('RGB', (self.video_width, self.video_height), (30,30,30))
        ctk_img = ctk.CTkImage(light_image=frame_black, dark_image=frame_black, size=(self.video_width, self.video_height))
        self.label_video.configure(image=ctk_img, text="Cámara Apagada")
        
        self.label_detected_sign.configure(text="---")
        self.btn_toggle_camera.configure(text="Iniciar Cámara")
        print("[INFO] Cámara detenida.")

    def gtts_worker_method(self):
        """Worker que consume de gtts_queue, genera audio y vacía cola de reproducción."""
        while True:
            text = self.gtts_queue.get()
            if text is None: break
            
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except Empty:
                    continue
                audio_queue.task_done()
                
            generate_audio_file(text)
            self.gtts_queue.task_done()

    def update_gui_loop(self):
        """Bucle en HILO PRINCIPAL para actualizar la GUI de forma segura."""
        if not self.is_camera_on: return

        # 1. Actualizar el frame de video
        with self.frame_lock:
            img_pil = self.current_frame
            
        if img_pil:
            img_resized = img_pil.resize((self.video_width, self.video_height), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(self.video_width, self.video_height))
            self.label_video.configure(image=ctk_img, text="")

        # 2. Actualizar el texto de la seña (leyendo el estado)
        current_time = time.time()
        display_text = "---"
        
        if self.current_sign_text:
            if current_time - self.sign_display_start_time > SIGN_DISPLAY_DURATION:
                self.current_sign_text = "" # Limpiar estado
            else:
                display_text = self.current_sign_text.upper() # Usar estado
        
        self.label_detected_sign.configure(text=display_text)

        # 3. Re-calendar la actualización
        self.update_id = self.after(20, self.update_gui_loop) # ~50 FPS

    def video_processing_loop(self):
        """HILO 1: Captura de Cámara y MediaPipe (Productor de Keypoints)"""
        while self.is_camera_on and self.cap:
            ret, frame = self.cap.read()
            if not ret: 
                # Si 'ret' es falso, la cámara se desconectó o se liberó (release)
                break # Salir del bucle
            
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            results = self.holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            draw_styled_landmarks(frame, results)
            
            keypoints = extract_keypoints(results)
            try:
                self.keypoint_queue.put_nowait(keypoints)
            except Full:
                pass 
                
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with self.frame_lock:
                self.current_frame = img_pil
        
        print("[Debug] Hilo de video terminado.")

    def prediction_loop(self):
        """HILO 2: Predicción del Modelo (Consumidor de Keypoints)"""
        while self.is_camera_on:
            try:
                keypoints = self.keypoint_queue.get(timeout=1.0)
                
                # --- CAMBIO CLAVE: Usar 'break' ---
                if keypoints is None: # Señal de parada
                    break # Salir del bucle
                    
            except Empty:
                continue # Timeout, volver a chequear self.is_camera_on
                
            # 2. Construir la secuencia
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-SEQUENCE_LENGTH:]
            
            # 3. Ejecutar la predicción (el paso lento)
            if len(self.sequence) == SEQUENCE_LENGTH and self.modelo:
                res = self.modelo.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                
                if np.max(res) > PREDICTION_THRESHOLD:
                    predicted_sign = self.actions[np.argmax(res)]
                    current_time = time.time()
                    
                    if current_time - self.last_prediction_time > PREDICTION_COOLDOWN:
                        if predicted_sign != self.current_sign_text:
                            
                            self.current_sign_text = predicted_sign
                            self.sign_display_start_time = current_time
                            self.gtts_queue.put(self.current_sign_text)
                            
                        self.last_prediction_time = current_time
            
            self.keypoint_queue.task_done()
        
        print("[Debug] Hilo de predicción terminado.")


    def on_closing(self):
        """Llamado cuando el usuario cierra la ventana (clic en 'X' o botón Salir)."""
        print("Cerrando aplicación...")
        
        # 1. Detener la cámara si está encendida
        if self.is_camera_on:
            self.stop_camera()
        
        # 2. Detener el bucle de la GUI (por si la cámara estaba apagada)
        if self.update_id:
            self.after_cancel(self.update_id)
            self.update_id = None
            
        # 3. Programar el apagado final
        self.after(50, self._final_shutdown)

    def _final_shutdown(self):
        """Apagado final y limpio."""
        
        # Señal de parada para los hilos de audio
        self.gtts_queue.put(None)
        audio_queue.put(None)
        
        # Limpiar recursos finales
        self.holistic.close()
        
        # Destruir la ventana
        print("[INFO] Aplicación cerrada.")
        self.destroy()

# --- EJECUTAR APP ---
if __name__ == "__main__":
    app = NeuroLESSAApp()
    app.mainloop()