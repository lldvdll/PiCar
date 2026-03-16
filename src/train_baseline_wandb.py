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
WANDB_ENTITY = "lpxdv2-university-of-nottingham"  

CONFIG = {
    "EXPERIMENT_NAME": "09_mlis2_test",
    "DESCRIPTION": "Test model on MLiS servers with GPUs",
    "OVERWRITE_EXPERIMENT": True,
    
    # --- Data Paths ---
    "TRAIN_CSV": os.path.join("data", "train_clean_weighted.csv"),
    "TRAIN_IMG_DIR": os.path.join("data", "training_data", "training_data"),
    "TEST_IMG_DIR": os.path.join("data", "test_data", "test_data"),
    "SUBMISSION_TEMPLATE": os.path.join("data", "sample_submission.csv"),
    "BAD_IMG_CSV": os.path.join("data", "bad_images.csv"),
    
    # --- Image Preprocessing ---
    "IMG_WIDTH_TARGET": 160,  
    "IMG_HEIGHT_TARGET": 96,
    "CROP_TOP_PIXELS": 110, 
    "CROP_BOTTOM_PIXELS": 30, 
    "CHANNELS": 3,
    
# --- Data Augmentation ---
    "AUG_USE_AUGMENTATION": True,
    "AUG_BRIGHTNESS_DELTA": 0.1,   
    "AUG_CONTRAST_LOWER": 0.9,     
    "AUG_CONTRAST_UPPER": 1.1,     
    "AUG_SATURATION_LOWER": 0.9,
    "AUG_SATURATION_UPPER": 1.1,
    "AUG_HUE_DELTA": 0.05,         
    "AUG_NOISE_STDDEV": 0.005,
    "AUG_ROTATION_FACTOR": 0.005,
    "AUG_TILT_FACTOR": 0.01,
    "AUG_CUTOUT_PROB": 0.5,        # 50% chance to apply cutout
    "AUG_CUTOUT_MIN_PIX": 10,      # Minimum mask size
    "AUG_CUTOUT_MAX_PIX": 50,      # Maximum mask size
    
    # --- Two-Phase Training Hyperparameters ---
    "EPOCHS_WARMUP": 5,             # Train frozen base with high LR
    "EPOCHS_FINETUNE":0,          # Train unfrozen base with low LR
    "LEARNING_RATE_WARMUP": 1e-3,
    "LEARNING_RATE_FINETUNE": 1e-4, 
    "BATCH_SIZE": 32,
    "OPTIMIZER": "adam",
    "LOSS_FUNCTION": "mse",
    
    # --- Model Architecture ---
    "BASE_MODEL": "MobileNetV2",
    "BASE_WEIGHTS": "imagenet",
    "UNFREEZE_TOP_N_LAYERS": 0,    # Set to 0 to skip fine-tuning entirely
    
    # --- Attention Head ---
    "USE_ATTENTION_BLOCK": True,
    "ATTN_BOTTLENECK_CHANNELS": 128, # Reduces 1280 channels down to this before attention
    
    # --- Head Flexibility Toggle ---
    "DENSE_UNITS_1": 256,       # 256
    "DENSE_UNITS_2": 128,       # 128
    "DROPOUT_RATE": 0.3,        
    "ACTIVATION_HIDDEN": "relu",
    "ACTIVATION_OUTPUT": "sigmoid"
}

# Derived Input Shape based on config
CONFIG["INPUT_SHAPE"] = (
    CONFIG["IMG_HEIGHT_TARGET"], 
    CONFIG["IMG_WIDTH_TARGET"], 
    CONFIG["CHANNELS"]
)
# ==============================================================================

def cleanup_existing_experiment(config, exp_dir):
    print(f"\n[WARNING] Overwrite is True. Cleaning up old '{config['EXPERIMENT_NAME']}' data...")
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
        print(f"  -> Deleted local directory: {exp_dir}")
        
    api = wandb.Api()
    try:
        path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
        runs = api.runs(path, filters={"display_name": config["EXPERIMENT_NAME"]})
        for run in runs:
            run.delete()
            print(f"  -> Deleted W&B cloud run: {run.id}")
    except Exception as e:
        pass

    log_path = os.path.join("experiments", "model_log.csv")
    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            clean_df = df[df["Experiment_Name"] != config["EXPERIMENT_NAME"]]
            clean_df.to_csv(log_path, index=False)
            print(f"  -> Removed old entries from {log_path}")
        except Exception:
            pass
    print("[INFO] Cleanup complete.\n")
    
def append_to_model_log(config, best_epoch, best_val_loss, wandb_run_id):
    log_path = os.path.join("experiments", "model_log.csv")
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
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

# --- 2. DATA PIPELINE ---

def reshape_image(img):
    """ Crop and resize image"""    
    top = CONFIG.get("CROP_TOP_PIXELS", 0)
    bottom_crop = CONFIG.get("CROP_BOTTOM_PIXELS", 0)
    bottom = -bottom_crop if bottom_crop > 0 else None
    img = img[top:bottom, :, :]
    img = tf.image.resize(img, [CONFIG["IMG_HEIGHT_TARGET"], CONFIG["IMG_WIDTH_TARGET"]])
    return img

# Instantiate Keras layers globally so tf.data doesn't recreate them on every image
random_rotation_layer = tf.keras.layers.RandomRotation(factor=CONFIG["AUG_ROTATION_FACTOR"], fill_mode='nearest')
random_translation_layer = tf.keras.layers.RandomTranslation(height_factor=CONFIG["AUG_TILT_FACTOR"], width_factor=0.0, fill_mode='nearest')
def augment_image(img):
    """ Image Augmentation 
        - colour and lighting: brightness, contrast, saturation, hue
        - image quality: add gausian noise
        - camera jitter: add small rotation
    """
    img = tf.image.random_brightness(img, max_delta=CONFIG["AUG_BRIGHTNESS_DELTA"])
    img = tf.image.random_contrast(img, lower=CONFIG["AUG_CONTRAST_LOWER"], upper=CONFIG["AUG_CONTRAST_UPPER"])
    img = tf.image.random_saturation(img, lower=CONFIG["AUG_SATURATION_LOWER"], upper=CONFIG["AUG_SATURATION_UPPER"])
    img = tf.image.random_hue(img, max_delta=CONFIG["AUG_HUE_DELTA"])
    
    if CONFIG["AUG_NOISE_STDDEV"] > 0:
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=CONFIG["AUG_NOISE_STDDEV"])
        img = img + noise
        
    # Apply spatial transforms using the globally instantiated layers
    if CONFIG["AUG_ROTATION_FACTOR"] > 0:
        img = tf.expand_dims(img, 0)
        img = random_rotation_layer(img, training=True) # Call the global layer here
        img = tf.squeeze(img, 0)
        
    if CONFIG.get("AUG_TILT_FACTOR", 0) > 0:
        img = tf.expand_dims(img, 0) # Needs batch dimension again
        img = random_translation_layer(img, training=True) # Call the global layer here
        img = tf.squeeze(img, 0)

    return tf.clip_by_value(img, 0.0, 1.0)


def random_cutout(img, probability, min_pixels, max_pixels):
    """Randomly masks a rectangular region of the image with black pixels."""
    def _apply_cutout(img):
        shape = tf.shape(img)
        H, W = shape[0], shape[1]

        # Random mask dimensions
        h = tf.random.uniform([], minval=min_pixels, maxval=max_pixels, dtype=tf.int32)
        w = tf.random.uniform([], minval=min_pixels, maxval=max_pixels, dtype=tf.int32)

        # Random anchor position (top-left corner)
        y = tf.random.uniform([], minval=0, maxval=tf.maximum(1, H - h), dtype=tf.int32)
        x = tf.random.uniform([], minval=0, maxval=tf.maximum(1, W - w), dtype=tf.int32)

        # Create a boolean mask grid
        yy, xx = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
        mask_y = tf.logical_and(yy >= y, yy < y + h)
        mask_x = tf.logical_and(xx >= x, xx < x + w)
        mask = tf.expand_dims(tf.logical_and(mask_y, mask_x), -1)

        # Apply mask (replace True values with 0.0)
        return tf.where(mask, tf.zeros_like(img), img)

    # Only apply if we beat the random probability
    do_cutout = tf.random.uniform([]) < probability
    return tf.cond(do_cutout, lambda: _apply_cutout(img), lambda: img)

def preprocess_image(image_path, angle=None, speed=None, augment=False):
    """
        Normalise image between 0 and 1.
        Crop top of image to remove background
        Resize image
        Augment (training only)
        Return image and labels if training, image only if inference
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=CONFIG["CHANNELS"])
    img = tf.cast(img, tf.float32) / 255.0
    
    if augment and CONFIG.get("AUG_USE_AUGMENTATION", False):
        img = augment_image(img)
        
    if augment and CONFIG.get("AUG_USE_AUGMENTATION", False):
        img = random_cutout(
            img, 
            probability=CONFIG["AUG_CUTOUT_PROB"], 
            min_pixels=CONFIG["AUG_CUTOUT_MIN_PIX"], 
            max_pixels=CONFIG["AUG_CUTOUT_MAX_PIX"]
        )
    
    img = reshape_image(img)
        
    if angle is not None and speed is not None:
        return img, {'angle_output': angle, 'speed_output': speed}
    
    return img

def prepare_data_pipelines():
    """
        Prepare training and validation data pipelines
        Load image file log
        Ignore any flagged images from data/bad_images.csv
        Sample from dataset using inverse weights - de-biasing data
        Split into train/val
        ...do tf stuf??
        Set batch size
        Return training and valudation sets
    """
    df = pd.read_csv(CONFIG["TRAIN_CSV"])
    
    # Drop flagged images
    bad_df = pd.read_csv(CONFIG["BAD_IMG_CSV"])
    bad_list = bad_df['filename'].astype(str).tolist()
    df['check_name'] = df['image_id'].astype(float).astype(int).astype(str) + '.png'
    initial_count = len(df)
    df = df[~df['check_name'].isin(bad_list)].drop(columns=['check_name'])
    print(f"[INFO] Dropped {initial_count - len(df)} manually flagged bad images.")
    
    # NEW: The Magic Balancing Act
    # We mathematically sample exactly the length of the dataset, but using our inverse weights.
    # replace=True means rare sharp-corners will be duplicated, while the thousands of 
    # straight-line images will be heavily ignored.
    balanced_df = df.sample(n=len(df), replace=True, weights='sample_weight', random_state=42)
    
    # Split the fully balanced dataset
    train_df, val_df = train_test_split(balanced_df, test_size=0.2, random_state=42)
    
    def create_ds(dataframe, is_training):
        paths = dataframe['filepath'].values
        angles = dataframe['angle'].values.astype(np.float32)
        speeds = dataframe['speed'].values.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, angles, speeds))
        ds = ds.map(lambda p, a, s: preprocess_image(p, a, s, augment=is_training), num_parallel_calls=tf.data.AUTOTUNE)
        if is_training:
            ds = ds.shuffle(buffer_size=len(dataframe))
        return ds.batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)
    
    return create_ds(train_df, is_training=True), create_ds(val_df, is_training=False)

# --- 3. MODEL ARCHITECTURE ---
def build_initial_model():
    """
        Builds the neural network architecture
            Load MobileNetV2 base model
            Freeze weights (they will be unfrozen later after head warm up epochs)
                        
    """
    inputs = tf.keras.Input(shape=CONFIG["INPUT_SHAPE"])
    base_model = MobileNetV2(input_shape=CONFIG["INPUT_SHAPE"], include_top=False, weights=CONFIG["BASE_WEIGHTS"])
    
    # Phase 1: Base is completely frozen
    base_model.trainable = False 
    features = base_model(inputs, training=False)
    
    # --- OPTIONAL ATTENTION BLOCK ---
    if CONFIG.get("USE_ATTENTION_BLOCK", False):
        print("[INFO] Building Spatial Attention Block...")
        # 1. Bottleneck: Reduce channel depth to save compute (1280 -> 128)
        reduced_features = layers.Conv2D(CONFIG["ATTN_BOTTLENECK_CHANNELS"], kernel_size=1, activation="relu")(features)
        
        # 2. Attention Map: Create a 1-channel spatial heatmap (values 0.0 to 1.0)
        attention_map = layers.Conv2D(1, kernel_size=1, activation="sigmoid")(reduced_features)
        
        # 3. Apply Attention: Multiply the ORIGINAL features (1280) by the heatmap (1)
        weighted_features = layers.Multiply()([features, attention_map])
        
        # Pass the weighted features into the GAP layer instead of the raw features
        x = layers.GlobalAveragePooling2D()(weighted_features)
    else:
        # Standard flow without attention
        x = layers.GlobalAveragePooling2D()(features)
    # --------------------------------
    
    # --- FLEXIBLE HEAD ARCHITECTURE ---
    print("[INFO] Building DEEP 1-Layer Head with Dropout...")
    x = layers.Dense(CONFIG["DENSE_UNITS_1"], activation=CONFIG["ACTIVATION_HIDDEN"])(x)
    x = layers.Dropout(CONFIG["DROPOUT_RATE"])(x)
    if CONFIG["DENSE_UNITS_2"] is not None:
        print("[INFO] Building DEEP 2-Layer Head with Dropout...")
        x = layers.Dense(CONFIG["DENSE_UNITS_2"], activation=CONFIG["ACTIVATION_HIDDEN"])(x)
        x = layers.Dropout(CONFIG["DROPOUT_RATE"])(x)
    # ----------------------------------
    
    angle_out = layers.Dense(1, activation=CONFIG["ACTIVATION_OUTPUT"], name='angle_output')(x)
    speed_out = layers.Dense(1, activation=CONFIG["ACTIVATION_OUTPUT"], name='speed_output')(x)
    
    model = models.Model(inputs=inputs, outputs=[angle_out, speed_out])
    
    opt = optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE_WARMUP"])
    model.compile(optimizer=opt,
                  loss={'angle_output': CONFIG["LOSS_FUNCTION"], 'speed_output': CONFIG["LOSS_FUNCTION"]},
                  metrics={'angle_output': 'mae', 'speed_output': 'mae'})
                  
    return model, base_model

# --- 4. MAIN ORCHESTRATOR ---
def main():
    print(f"\n========== STARTING EXPERIMENT: {CONFIG['EXPERIMENT_NAME']} ==========")
    
    exp_dir = os.path.join("experiments", CONFIG["EXPERIMENT_NAME"])
    if os.path.exists(exp_dir):
        if CONFIG.get("OVERWRITE_EXPERIMENT", False):
            cleanup_existing_experiment(CONFIG, exp_dir)
        else:
            raise FileExistsError("Experiment exists. Change name or set OVERWRITE_EXPERIMENT to True.")
            
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "experiment_details.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)
    
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=CONFIG["EXPERIMENT_NAME"], 
               dir=exp_dir, config=CONFIG, notes=CONFIG["DESCRIPTION"])
    
    os.makedirs("models", exist_ok=True)
    train_ds, val_ds = prepare_data_pipelines()
    
    # Extract both the compiled model and the base_model so we can unfreeze it later
    model, base_model = build_initial_model() 
    
    callbacks = [
        WandbMetricsLogger(),
        ModelCheckpoint(filepath=os.path.join(exp_dir, "best_model.h5"), monitor="val_loss", save_best_only=True, verbose=1)
    ]
    
    start_time = time.time()
    
    # ------------------------------------------------------------------------------------
    # PHASE 1: WARM-UP (Frozen Base, High LR)
    # ------------------------------------------------------------------------------------
    print(f"\n[INFO] --- PHASE 1: WARM-UP ({CONFIG['EPOCHS_WARMUP']} Epochs) ---")
    history_1 = model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_WARMUP"], callbacks=callbacks)
    
    # Store history so we can find the best epoch across both phases
    combined_val_loss = history_1.history['val_loss']
    
    # ------------------------------------------------------------------------------------
    # PHASE 2: FINE-TUNING (Unfrozen Base, Low LR)
    # ------------------------------------------------------------------------------------
    if CONFIG["UNFREEZE_TOP_N_LAYERS"] > 0 and CONFIG["EPOCHS_FINETUNE"] > 0:
        print(f"\n[INFO] --- PHASE 2: FINE-TUNING TOP {CONFIG['UNFREEZE_TOP_N_LAYERS']} LAYERS ({CONFIG['EPOCHS_FINETUNE']} Epochs) ---")
        
        base_model.trainable = True
        for layer in base_model.layers[:-CONFIG["UNFREEZE_TOP_N_LAYERS"]]:
            layer.trainable = False
            
        # We MUST recompile the model for the trainability changes to take effect
        opt = optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE_FINETUNE"])
        model.compile(optimizer=opt,
                      loss={'angle_output': CONFIG["LOSS_FUNCTION"], 'speed_output': CONFIG["LOSS_FUNCTION"]},
                      metrics={'angle_output': 'mae', 'speed_output': 'mae'})
        
        total_epochs = CONFIG["EPOCHS_WARMUP"] + CONFIG["EPOCHS_FINETUNE"]
        
        # Notice we use `initial_epoch` so Keras/W&B knows we are continuing where we left off
        history_2 = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=total_epochs, 
            initial_epoch=CONFIG["EPOCHS_WARMUP"], 
            callbacks=callbacks
        )
        combined_val_loss.extend(history_2.history['val_loss'])

    # ------------------------------------------------------------------------------------
    # EVALUATION & LOGGING
    # ------------------------------------------------------------------------------------
    training_time = (time.time() - start_time) / 60
    best_epoch_index = int(np.argmin(combined_val_loss))
    best_epoch_human = best_epoch_index + 1
    best_val_loss = float(combined_val_loss[best_epoch_index])
    
    append_to_model_log(CONFIG, best_epoch_human, best_val_loss, wandb.run.id)
    
    wandb.run.summary["training_time_minutes"] = round(training_time, 2)
    wandb.run.summary["best_epoch"] = best_epoch_human
    wandb.run.summary["best_val_loss"] = best_val_loss
    
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
    wandb.finish()
    
    print(f"[INFO] Submission saved to {submission_file}")
    print("======================================================================\n")

if __name__ == "__main__":
    main()