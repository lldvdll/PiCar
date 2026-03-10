import os
import json
import time
import csv
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# W&B Imports
import wandb
from wandb.integration.keras import WandbMetricsLogger

# ==============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION (All your dials in one place)
# ==============================================================================
WANDB_PROJECT = "PiCar"
WANDB_ENTITY = "lpxdv2-university-of-nottingham"  # Change this to your W&B Team name!

CONFIG = {
    "EXPERIMENT_NAME": "02_wider_mlp_with_dropout",
    "DESCRIPTION": "Frozen base. Wider MLP head (256 -> 128) with 30% Dropout to prevent overfitting.",
    "OVERWRITE_EXPERIMENT": True,
    
    # --- Data Paths ---
    "TRAIN_CSV": os.path.join("data", "train.csv"),
    "TRAIN_IMG_DIR": os.path.join("data", "training_data", "training_data"),
    "TEST_IMG_DIR": os.path.join("data", "test_data", "test_data"),
    "SUBMISSION_TEMPLATE": os.path.join("data", "sample_submission.csv"),
    
    # --- Image Preprocessing ---
    "IMG_WIDTH_TARGET": 80,
    "IMG_HEIGHT_TARGET": 60,
    "CROP_TOP_PIXELS": 15, # Crops top 15px. Remaining height = 45px
    "CHANNELS": 3,
    
    # --- Training Hyperparameters ---
    "EPOCHS": 25,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.001,
    "OPTIMIZER": "adam",
    "LOSS_FUNCTION": "mse",
    
    # --- Model Architecture ---
    "BASE_MODEL": "MobileNetV2",
    "BASE_WEIGHTS": "imagenet",
    "UNFREEZE_TOP_N_LAYERS": 0, # Set to 0 to keep completely frozen
    "DENSE_UNITS_1": 256,       # First, wider layer
    "DENSE_UNITS_2": 128,       # Second, extra layer
    "DROPOUT_RATE": 0.3,        # Drops 30% of connections to prevent overfitting
    "ACTIVATION_HIDDEN": "relu",
    "ACTIVATION_OUTPUT": "sigmoid"
}

# Derived Input Shape based on config
CONFIG["INPUT_SHAPE"] = (
    CONFIG["IMG_HEIGHT_TARGET"] - CONFIG["CROP_TOP_PIXELS"], 
    CONFIG["IMG_WIDTH_TARGET"], 
    CONFIG["CHANNELS"]
)
# ==============================================================================

def cleanup_existing_experiment(config, exp_dir):
    """Deletes local folder, removes CSV entry, and deletes W&B cloud run."""
    print(f"\n[WARNING] Overwrite is True. Cleaning up old '{config['EXPERIMENT_NAME']}' data...")
    
    # 1. Delete Local Directory
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
        print(f"  -> Deleted local directory: {exp_dir}")
        
    # 2. Delete from W&B Cloud
    api = wandb.Api()
    try:
        # Search for runs with this exact name in your project
        path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
        runs = api.runs(path, filters={"display_name": config["EXPERIMENT_NAME"]})
        for run in runs:
            run.delete()
            print(f"  -> Deleted W&B cloud run: {run.id}")
    except Exception as e:
        print(f"  -> [Note] Could not delete W&B run (may not exist or permission issue): {e}")

    # 3. Remove old entry from the CSV log
    log_path = os.path.join("experiments", "model_log.csv")
    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            # Filter out the old run, save the rest back to the CSV
            clean_df = df[df["Experiment_Name"] != config["EXPERIMENT_NAME"]]
            clean_df.to_csv(log_path, index=False)
            print(f"  -> Removed old entries from {log_path}")
        except Exception as e:
            pass
    print("[INFO] Cleanup complete.\n")
    
    
def append_to_model_log(config, best_epoch, best_val_loss, wandb_run_id):
    """Appends experiment results to a central CSV log."""
    log_path = os.path.join("experiments", "model_log.csv")
    file_exists = os.path.isfile(log_path)
    
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write headers if the file is brand new
        if not file_exists:
            writer.writerow(["Date", "Experiment_Name", "WandB_ID", "Best_Epoch", "Best_Val_Loss", "Description"])
            
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            config["EXPERIMENT_NAME"],
            wandb_run_id,
            best_epoch,
            round(best_val_loss, 5),
            config["DESCRIPTION"]
        ])
    print(f"[INFO] Experiment logged to {log_path}")

# --- 2. DATA PIPELINE ---
def validate_image_paths(df, img_dir):
    """Checks files for existence AND fixes the .0 float issue from Pandas."""
    df = df.dropna(subset=['image_id']) 
    valid_paths, valid_indices = [], []
    
    for idx, row in df.iterrows():
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
        raise ValueError("[FATAL ERROR] All images were missing! Check your paths.")
    return valid_paths, clean_df

def preprocess_image(image_path, angle=None, speed=None):
    """Resizes and crops based on CONFIG."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=CONFIG["CHANNELS"])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, [CONFIG["IMG_HEIGHT_TARGET"], CONFIG["IMG_WIDTH_TARGET"]])
    img = img[CONFIG["CROP_TOP_PIXELS"]:, :, :]
    
    if angle is not None and speed is not None:
        return img, {'angle_output': angle, 'speed_output': speed}
    return img

def prepare_data_pipelines():
    df = pd.read_csv(CONFIG["TRAIN_CSV"])
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_paths, train_df = validate_image_paths(train_df, CONFIG["TRAIN_IMG_DIR"])
    val_paths, val_df = validate_image_paths(val_df, CONFIG["TRAIN_IMG_DIR"])
    
    def create_ds(paths, dataframe, shuffle):
        angles = dataframe['angle'].values.astype(np.float32)
        speeds = dataframe['speed'].values.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, angles, speeds))
        ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        return ds.batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)

    return create_ds(train_paths, train_df, shuffle=True), create_ds(val_paths, val_df, shuffle=False)

# --- 3. MODEL ARCHITECTURE ---
def build_compile_model():
    inputs = tf.keras.Input(shape=CONFIG["INPUT_SHAPE"])
    
    base_model = MobileNetV2(
        input_shape=CONFIG["INPUT_SHAPE"], 
        include_top=False, 
        weights=CONFIG["BASE_WEIGHTS"]
    )
    
    # FIX #1: Replaced broken FREEZE_BASE_MODEL logic with actual unfreezing logic
    if CONFIG["UNFREEZE_TOP_N_LAYERS"] > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-CONFIG["UNFREEZE_TOP_N_LAYERS"]]:
            layer.trainable = False
        print(f"[INFO] Unfroze the top {CONFIG['UNFREEZE_TOP_N_LAYERS']} layers of {CONFIG['BASE_MODEL']}.")
    else:
        base_model.trainable = False
        print(f"[INFO] {CONFIG['BASE_MODEL']} is completely frozen.")
    
    features = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(features)
    
    # --- NEW WIDER, DEEPER HEAD ---
    # First Block
    x = layers.Dense(CONFIG["DENSE_UNITS_1"], activation=CONFIG["ACTIVATION_HIDDEN"])(x)
    x = layers.Dropout(CONFIG["DROPOUT_RATE"])(x)
    
    # Second Block
    x = layers.Dense(CONFIG["DENSE_UNITS_2"], activation=CONFIG["ACTIVATION_HIDDEN"])(x)
    x = layers.Dropout(CONFIG["DROPOUT_RATE"])(x)
    # ------------------------------
    
    angle_out = layers.Dense(1, activation=CONFIG["ACTIVATION_OUTPUT"], name='angle_output')(x)
    speed_out = layers.Dense(1, activation=CONFIG["ACTIVATION_OUTPUT"], name='speed_output')(x)
    
    model = models.Model(inputs=inputs, outputs=[angle_out, speed_out])
    
    opt = optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE"])
    model.compile(optimizer=opt,
                  loss={'angle_output': CONFIG["LOSS_FUNCTION"], 'speed_output': CONFIG["LOSS_FUNCTION"]},
                  metrics={'angle_output': 'mae', 'speed_output': 'mae'})
    return model

# --- 4. MAIN ORCHESTRATOR ---
def main():
    print(f"\n========== STARTING EXPERIMENT: {CONFIG['EXPERIMENT_NAME']} ==========")
    
    # 1. Unified Directory & Overwrite Protection
    exp_dir = os.path.join("experiments", CONFIG["EXPERIMENT_NAME"])
    
    if os.path.exists(exp_dir):
        if CONFIG.get("OVERWRITE_EXPERIMENT", False):
            cleanup_existing_experiment(CONFIG, exp_dir)
        else:
            raise FileExistsError(
                f"\n[FATAL ERROR] The experiment '{CONFIG['EXPERIMENT_NAME']}' already exists!\n"
                f"Set 'OVERWRITE_EXPERIMENT': True in CONFIG to overwrite, or pick a new name."
            )
            
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save the config JSON right into this new folder
    config_file = os.path.join(exp_dir, "experiment_details.json")
    with open(config_file, "w") as f:
        json.dump(CONFIG, f, indent=4)
    
    # 1. Initialize Weights & Biases
    wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY, 
            name=CONFIG["EXPERIMENT_NAME"], # Forces W&B Display Name to match
            dir=exp_dir,                    # Forces W&B to save files inside our unified folder
            config=CONFIG,
            notes=CONFIG["DESCRIPTION"]
        )
    
    os.makedirs("models", exist_ok=True)
    train_ds, val_ds = prepare_data_pipelines()
    model = build_compile_model()
    
    # 2. Callbacks (W&B Logger + Local Checkpoint)
    callbacks = [
        WandbMetricsLogger(),
        ModelCheckpoint(
            filepath=os.path.join(exp_dir, "best_model.h5"), # Saved inside unified folder
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 3. Train
    print("[INFO] Training model...")
    start_time = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS"], callbacks=callbacks)
    
    # FIX #2: Cleaned up the epoch extraction logic to prevent IndexErrors
    training_time = (time.time() - start_time) / 60
    best_epoch_index = int(np.argmin(history.history['val_loss']))
    best_epoch_human = best_epoch_index + 1
    best_val_loss = float(history.history['val_loss'][best_epoch_index])
    
    # NEW: Log to our central CSV
    append_to_model_log(CONFIG, best_epoch_human, best_val_loss, wandb.run.id)
    
    wandb.run.summary["training_time_minutes"] = round(training_time, 2)
    wandb.run.summary["best_epoch"] = best_epoch_human
    wandb.run.summary["best_val_loss"] = best_val_loss
    
    # 4. Generate Submission
    print("\n[INFO] Generating Kaggle submission...")
    best_model = tf.keras.models.load_model(os.path.join(exp_dir, "best_model.h5"))
    sub_df = pd.read_csv(CONFIG["SUBMISSION_TEMPLATE"])
    
    test_paths, sub_df = validate_image_paths(sub_df, CONFIG["TEST_IMG_DIR"])
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(CONFIG["BATCH_SIZE"])
    
    predictions = best_model.predict(test_ds)
    sub_df['angle'] = predictions[0].flatten()
    sub_df['speed'] = predictions[1].flatten()
    
    submission_file = os.path.join(exp_dir, "submission.csv")
    sub_df.to_csv(submission_file, index=False)
    
    # Tell W&B we are done tracking this run
    wandb.finish()
    
    print(f"[INFO] Submission saved to {submission_file}")
    print("======================================================================\n")

if __name__ == "__main__":
    main()