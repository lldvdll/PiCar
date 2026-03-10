import os
import time
import datetime
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
CONFIG = {
    "EXPERIMENT_NAME": "01_baseline_mobilenet_sigmoid",
    "DESCRIPTION": "Initial baseline. Modularized pipeline with strict path validation.",
    "DATA_DIR": "data",
    "TRAIN_CSV": os.path.join("data", "train.csv"),
    "TRAIN_IMG_DIR": os.path.join("data", "training_data", "training_data"),
    "TEST_IMG_DIR": os.path.join("data", "test_data", "test_data"),
    "SUBMISSION_TEMPLATE": os.path.join("data", "sample_submission.csv"),
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "INPUT_SHAPE": (45, 80, 3) 
}

# --- 2. LOGGING & UTILITIES ---
def setup_experiment_logging(config):
    """Creates directories and saves config details."""
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "fit", f"{config['EXPERIMENT_NAME']}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "experiment_details.json")
    with open(log_file, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"[INFO] Experiment details saved to {log_file}")
    return log_dir

def update_training_time(log_dir, config, training_time):
    """Updates the JSON log with the total training time."""
    config["TRAINING_TIME_MINUTES"] = round(training_time / 60, 2)
    log_file = os.path.join(log_dir, "experiment_details.json")
    with open(log_file, "w") as f:
        json.dump(config, f, indent=4)

# --- 3. DATA PIPELINE & VALIDATION ---
def validate_image_paths(df, img_dir):
    """Checks files for existence AND fixes the .0 float issue from Pandas."""
    df = df.dropna(subset=['image_id']) 
    
    valid_paths = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        # THE FIX: Safely convert float (2180.0) -> int (2180) -> string ('2180')
        filename = str(int(float(row['image_id'])))
        
        if not filename.endswith('.png'):
            filename += '.png'
            
        full_path = os.path.join(img_dir, filename).replace("\\", "/") 
        
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            valid_paths.append(full_path)
            valid_indices.append(idx)
        else:
            print(f"[WARNING] Skipping missing or 0-byte file: {full_path}")
            
    clean_df = df.loc[valid_indices].copy()
    
    if len(clean_df) == 0:
        raise ValueError("[FATAL ERROR] All images were missing or empty! Check your folder paths.")
        
    return valid_paths, clean_df

def preprocess_image(image_path, angle=None, speed=None):
    """Reads, resizes (80 width), crops (top 1/4), and normalizes."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, [60, 80])
    img = img[15:, :, :]
    
    if angle is not None and speed is not None:
        return img, {'angle_output': angle, 'speed_output': speed}
    return img

def prepare_data_pipelines(config):
    """Loads CSVs, validates paths, and builds train/val tf.data.Datasets."""
    df = pd.read_csv(config["TRAIN_CSV"])
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print("[INFO] Validating training image paths...")
    train_paths, train_df = validate_image_paths(train_df, config["TRAIN_IMG_DIR"])
    print("[INFO] Validating validation image paths...")
    val_paths, val_df = validate_image_paths(val_df, config["TRAIN_IMG_DIR"])
    
    def create_ds(paths, dataframe, shuffle):
        angles = dataframe['angle'].values.astype(np.float32)
        speeds = dataframe['speed'].values.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, angles, speeds))
        ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        return ds.batch(config["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)

    train_ds = create_ds(train_paths, train_df, shuffle=True)
    val_ds = create_ds(val_paths, val_df, shuffle=False)
    
    return train_ds, val_ds

# --- 4. MODEL ARCHITECTURE ---
def build_compile_model(input_shape):
    """Assembles the frozen MobileNetV2 base with custom sigmoid heads."""
    inputs = tf.keras.Input(shape=input_shape)
    
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    
    features = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(features)
    x = layers.Dense(64, activation='relu')(x)
    
    angle_out = layers.Dense(1, activation='sigmoid', name='angle_output')(x)
    speed_out = layers.Dense(1, activation='sigmoid', name='speed_output')(x)
    
    model = models.Model(inputs=inputs, outputs=[angle_out, speed_out])
    model.compile(optimizer='adam',
                  loss={'angle_output': 'mse', 'speed_output': 'mse'},
                  metrics={'angle_output': 'mae', 'speed_output': 'mae'})
    return model

# --- 5. TRAINING & EVALUATION ---
def get_callbacks(log_dir, model_name):
    """Sets up TensorBoard and Model Checkpointing."""
    tb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    chkpt = ModelCheckpoint(
        filepath=os.path.join("models", f"{model_name}_best.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    return [tb, chkpt]

def generate_submission(model_name, config):
    """Predicts on test data and outputs a Kaggle-ready CSV."""
    print("\n[INFO] Generating Kaggle submission...")
    best_model = tf.keras.models.load_model(os.path.join("models", f"{model_name}_best.h5"))
    sub_df = pd.read_csv(config["SUBMISSION_TEMPLATE"])
    
    print("[INFO] Validating test image paths...")
    test_paths, sub_df = validate_image_paths(sub_df, config["TEST_IMG_DIR"])
    
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(config["BATCH_SIZE"])
    
    predictions = best_model.predict(test_ds)
    sub_df['angle'] = predictions[0].flatten()
    sub_df['speed'] = predictions[1].flatten()
    
    submission_file = f"submission_{model_name}.csv"
    sub_df.to_csv(submission_file, index=False)
    print(f"[INFO] Submission saved to {submission_file}")

# --- 6. MAIN ORCHESTRATOR ---
def main():
    print(f"\n========== STARTING EXPERIMENT: {CONFIG['EXPERIMENT_NAME']} ==========")
    
    # 1. Setup
    log_dir = setup_experiment_logging(CONFIG)
    
    # 2. Data
    train_ds, val_ds = prepare_data_pipelines(CONFIG)
    
    # 3. Model & Callbacks
    model = build_compile_model(CONFIG["INPUT_SHAPE"])
    callbacks = get_callbacks(log_dir, CONFIG['EXPERIMENT_NAME'])
    
    # 4. Train
    print("[INFO] Training model...")
    start_time = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS"], callbacks=callbacks)
    update_training_time(log_dir, CONFIG, time.time() - start_time)
    
    # 5. Submit
    generate_submission(CONFIG['EXPERIMENT_NAME'], CONFIG)
    print("======================================================================\n")

if __name__ == "__main__":
    main()