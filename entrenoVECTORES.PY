import numpy as np
import tensorflow as tf
import os
import glob
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Masking, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision # Importación para precisión mixta
# --- 1. FUNCIONES DE AYUDA Y PREPROCESAMIENTO ---
def augment_data(sequence, noise_level=0.005):
    """Aplica ruido gaussiano a una secuencia para aumentar los datos."""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise
def load_sequences_and_labels(data_path, actions):
    """
    Crea una lista de todas las rutas a las secuencias y sus etiquetas correspondientes.
    No carga los datos en memoria, solo las rutas.
    """
    filepaths = []
    for action_idx, action in enumerate(actions):
        action_path = Path(data_path) / action
        sequence_paths = glob.glob(os.path.join(action_path, "*"))
        for sequence_path in sequence_paths:
            # Asegurarse de que el path no esté vacío
            if len(os.listdir(sequence_path)) > 0:
                filepaths.append((sequence_path, action_idx))
    return filepaths
def data_generator(filepaths, actions, frames_per_seq, expected_dim, is_training=False):
    """
    Generador de Python que carga y preprocesa una secuencia a la vez.
    Esto es clave para la eficiencia de memoria.
    """
    num_classes = len(actions)
    for sequence_path, label_idx in filepaths:
        frames_data = []
        frame_files = sorted(glob.glob(os.path.join(sequence_path, "*.npy")))
        for frame_file in frame_files:
            res = np.load(frame_file)
            frames_data.append(res)
        if not frames_data:
            continue
        sequence = np.array(frames_data)
        # --- Normalización y Relleno ---
        if len(sequence) < frames_per_seq:
            pad_width = frames_per_seq - len(sequence)
            padding = np.zeros((pad_width, sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        else:
            sequence = sequence[:frames_per_seq]
        if sequence.shape[1] < expected_dim:
            padding_hand = np.zeros((frames_per_seq, expected_dim - sequence.shape[1]))
            sequence = np.hstack([sequence, padding_hand])
        elif sequence.shape[1] > expected_dim:
            print(f"[AVISO] Formato inesperado, ignorando: {sequence_path}")
            continue
        # --- Aumento de Datos (Solo para entrenamiento) ---
        if is_training:
            sequence = augment_data(sequence)
            
        # Usar to_categorical para la etiqueta
        yield sequence.astype(np.float32), to_categorical(label_idx, num_classes=num_classes).astype(np.float32)
def build_model(num_classes, frames_per_seq, expected_dim):
    """
    Construye, compila y retorna el modelo de IA.
    Usa GRU para eficiencia y Dropout para precisión.
    """
    print("\n[INFO] Construyendo el modelo GRU...")
    model = Sequential([
        Input(shape=(frames_per_seq, expected_dim)),
        Masking(mask_value=0.0),
        # Usamos GRU: similar a LSTM pero más eficiente computacionalmente
        GRU(128, return_sequences=True, activation='tanh', recurrent_dropout=0.2),
        Dropout(0.4),
        GRU(256, return_sequences=True, activation='tanh', recurrent_dropout=0.2),
        Dropout(0.4),
        GRU(128, return_sequences=False, activation='tanh', recurrent_dropout=0.2),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model
def plot_history(history):
    """Visualiza el historial de entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.legend()
    ax1.set_title('Accuracy')
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.legend()
    ax2.set_title('Loss')
    plt.savefig('training_history.png')
    print("\n[INFO] Gráfico de historial guardado en 'training_history.png'")
    # plt.show() # <-- CORRECCIÓN: Eliminado para evitar el RuntimeError de Tkinter
    plt.close(fig) # <-- Buena práctica: Cierra la figura para liberar memoria
def evaluate_model(model, test_dataset, actions):
    """Muestra métricas de evaluación detalladas."""
    print("\n[INFO] Evaluando modelo...")
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_true_labels = np.argmax(y_true, axis=1)
    
    y_pred_probs = model.predict(test_dataset)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_true_labels, y_pred_labels, target_names=actions))
    print("\n--- Matriz de Confusión ---")
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    fig = plt.figure(figsize=(10, 8)) # <-- Captura la figura en una variable
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig('confusion_matrix.png')
    print("[INFO] Matriz de confusión guardada en 'confusion_matrix.png'")
    # plt.show() # <-- CORRECCIÓN: Eliminado para evitar el RuntimeError de Tkinter
    plt.close(fig) # <-- Buena práctica: Cierra la figura para liberar memoria
# --- 2. SCRIPT PRINCIPAL ---
def main(args):
    # 1. Asegura que las optimizaciones de OneDNN estén habilitadas
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
    # 2. Activa el compilador XLA (Accelerated Linear Algebra) para fusionar operaciones
    tf.config.optimizer.set_jit(True)
    # 3. Configuración de hilos para tu CPU i7-1185G7 (4 núcleos físicos / 8 hilos lógicos)
    # Usa los núcleos físicos para operaciones internas (cálculos matemáticos)
# 3. Configuración de hilos para tu CPU i3-1215U (6 núcleos / 8 hilos)
# Usa los núcleos de rendimiento y algunos de eficiencia para operaciones internas
    tf.config.threading.set_intra_op_parallelism_threads(6)
# Usa los 8 hilos lógicos para coordinación entre operaciones
    tf.config.threading.set_inter_op_parallelism_threads(8)
  
    mixed_precision.set_global_policy('mixed_bfloat16')
    print("\n[INFO] Optimizaciones de CPU (OneDNN, XLA, Precisión Mixta) activadas.")
    DATA_PATH = Path(args.data_path)
    MODEL_FOLDER = Path('modelo_entrenado')
    MODEL_FOLDER.mkdir(exist_ok=True)
    # Parámetros (obtenidos de argparse)
    FRAMES_PER_SEQUENCE = args.frames
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    # Dimensiones (asumiendo 2 manos de MediaPipe)
    EXPECTED_DIM = 21 * 3 * 2 # 126
    # --- Carga de Datos ---
    print(f"[INFO] Buscando gestos en: {DATA_PATH}")
    try:
        actions = sorted([d.name for d in DATA_PATH.iterdir() if d.is_dir()])
        num_classes = len(actions)
        if num_classes == 0:
            print(f"[ERROR] No se encontraron carpetas de gestos en '{DATA_PATH}'.")
            return
        print(f"[INFO] Gestos encontrados ({num_classes}): {actions}")
    except FileNotFoundError:
        print(f"[ERROR] La carpeta '{DATA_PATH}' no fue encontrada.")
        return
    # Guardar las etiquetas para uso futuro
    np.save(MODEL_FOLDER / "etiquetas.npy", actions)
    # Obtener rutas de archivos y dividirlas
    all_filepaths = load_sequences_and_labels(DATA_PATH, actions)
    if not all_filepaths:
        print(f"[ERROR] No se encontraron archivos de secuencias .npy en las carpetas de gestos.")
        return
        
    train_files, test_files = train_test_split(all_filepaths, test_size=0.20, random_state=42, stratify=[label for path, label in all_filepaths])
    print(f"\n[INFO] Total de muestras: {len(all_filepaths)}")
    print(f"Muestras de entrenamiento: {len(train_files)}")
    print(f"Muestras de prueba: {len(test_files)}")
    # --- Creación de `tf.data.Dataset` ---
    # Esto es el corazón de la eficiencia en CPU y memoria
    print("\n[INFO] Creando pipelines de datos eficientes con tf.data...")
    output_signature = (
        tf.TensorSpec(shape=(FRAMES_PER_SEQUENCE, EXPECTED_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
    )
    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(train_files, actions, FRAMES_PER_SEQUENCE, EXPECTED_DIM, is_training=True),
        output_signature=output_signature
    ).cache().shuffle(len(train_files)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(test_files, actions, FRAMES_PER_SEQUENCE, EXPECTED_DIM, is_training=False),
        output_signature=output_signature
    ).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    # --- Construcción y Entrenamiento ---
    model = build_model(num_classes, FRAMES_PER_SEQUENCE, EXPECTED_DIM)
    # Callbacks para un entrenamiento inteligente
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001, verbose=1)
    model_checkpoint_path = str(MODEL_FOLDER / "modelo_gestos.keras")
    model_checkpoint = ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    print("\n[INFO] Iniciando el entrenamiento...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )
    # --- Evaluación y Guardado ---
    print("\n[INFO] Entrenamiento completado.")
    print(f"\n[INFO] Cargando el mejor modelo desde: {model_checkpoint_path}")
    # Asegurarse de que el modelo cargado compile con las optimizaciones
    best_model = load_model(model_checkpoint_path)
    plot_history(history)
    evaluate_model(best_model, test_ds, actions)
    print(f"\n[✅] Modelo guardado en: '{MODEL_FOLDER}'")
    print("\n¡Proceso finalizado con éxito!")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento eficiente de modelo para gestos.')
    parser.add_argument('--data_path', type=str, default='data', help='Ruta a la carpeta de datos.')
    parser.add_argument('--epochs', type=int, default=300, help='Número máximo de épocas.')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del lote (batch size).')
    parser.add_argument('--frames', type=int, default=30, help='Número de frames por secuencia.')
    args = parser.parse_args()
    main(args)
