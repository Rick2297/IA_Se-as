import cv2
import numpy as np
import os
import mediapipe as mp
from pathlib import Path 
import time
# --- 1. FUNCIONES AUXILIARES ---
def draw_styled_landmarks(image, results):
    """Dibuja los landmarks de las manos con estilos personalizados y muestra el estado de detección."""
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    # Estilos para landmarks y conexiones (más finos)
    left_hand_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
    right_hand_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
    connection_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
    # Dibujar landmarks de ambas manos
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, left_hand_style, connection_style)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, right_hand_style, connection_style)
    # Mostrar texto de detección de manos
    if results.left_hand_landmarks or results.right_hand_landmarks:
        cv2.putText(image, "DETECCION OK", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(image, "SIN MANOS", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
def extract_keypoints(results):
    """Extrae las coordenadas de los landmarks. Devuelve un array de ceros si no se detecta una mano."""
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])
# --- 2. SCRIPT PRINCIPAL ---
def main():
    # --- Configuración ---
    DATA_PATH = Path('data')
    num_secuencias = 60
    frames_por_secuencia = 30
    # --- Inicialización ---
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return
    # --- Variables de Estado y Control ---
    state = 'ESPERANDO_GESTO' # Estados: ESPERANDO_GESTO, CUENTA_REGRESIVA, GRABANDO
    action_input = ""
    action_name = ""
    sequence_num = 0
    frame_num = 0
    countdown_timer = 0
    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            # --- Lógica de Estados ---
            if state == 'ESPERANDO_GESTO':
                prompt_text = f"Ingrese nombre del gesto: {action_input}"
                cv2.putText(frame, prompt_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "Presiona ENTER para iniciar o Q para salir.", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                key = cv2.waitKey(10) & 0xFF
                if key == 13 and action_input: # Enter
                    action_name = action_input
                    action_input = ""
                    sequence_num = 0
                    # Crear carpetas necesarias
                    (DATA_PATH / action_name).mkdir(parents=True, exist_ok=True)
                    print(f"\n[INFO] Gesto '{action_name}' configurado. Iniciando recolección.")
                    state = 'CUENTA_REGRESIVA'
                    countdown_timer = time.time()
                elif key == 8: # Backspace
                    action_input = action_input[:-1]
                elif key >= 97 and key <= 122: # A-Z (minúsculas)
                    action_input += chr(key)
                elif key == ord('q'):
                    break
            elif state == 'CUENTA_REGRESIVA':
                elapsed_time = time.time() - countdown_timer
                countdown = 3 - int(elapsed_time)
                if countdown > 0:
                    texto = f"Preparate para '{action_name}'... Empezando en {countdown}"
                    cv2.putText(frame, texto, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    state = 'GRABANDO'
                    frame_num = 0
                    print(f"[INFO] Grabando secuencia {sequence_num}...")
            elif state == 'GRABANDO':
                # Detección y dibujo de landmarks
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                draw_styled_landmarks(image, results)
                # Guardar keypoints solo si se detectan manos
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    keypoints = extract_keypoints(results)
                    sequence_path = DATA_PATH / action_name / str(sequence_num)
                    sequence_path.mkdir(exist_ok=True)
                    np.save(str(sequence_path / str(frame_num)), keypoints)
                frame_num += 1
                # Mostrar información
                info_texto = f"Gesto: {action_name} | Sec: {sequence_num} | Frame: {frame_num}"
                cv2.putText(image, info_texto, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                frame = image # Reemplazar el frame original con el procesado
                if frame_num >= frames_por_secuencia:
                    sequence_num += 1
                    if sequence_num >= num_secuencias:
                        print(f"[INFO] Recolección para '{action_name}' completada.")
                        state = 'ESPERANDO_GESTO'
                    else:
                        state = 'CUENTA_REGRESIVA'
                        countdown_timer = time.time()
            # Mostrar la ventana única
            cv2.imshow('Recoleccion de Datos', frame)
            # Lógica de salida general
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    # --- Finalización ---
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Proceso finalizado.")
if __name__ == '__main__':
    main()