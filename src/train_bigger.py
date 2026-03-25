import os
import json
import time
import csv
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
# --- KAGGLE HEAVYWEIGHT IMPORTS ---
from tensorflow.keras.applications import ConvNeXtBase, EfficientNetV2S, EfficientNetV2M
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import wandb
from wandb.integration.keras import WandbMetricsLogger

# ==============================================================================
# GPU SETUP
# ==============================================================================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] VRAM memory growth enabled. GPU is ready!\n")
    except RuntimeError as e:
        print(f"[ERROR] GPU Setup failed: {e}\n")

# ==============================================================================
# 1. KAGGLE CONFIGURATION 
# ==============================================================================
WANDB_PROJECT = "PiCar"
WANDB_ENTITY = "lpxdv2-university-of-nottingham"  

CONFIG = {
    "EXPERIMENT_NAME": "30_ConvNeXtBase",
    "DESCRIPTION": "Heavyweight architecture test for Kaggle generalisation.",
    "OVERWRITE_EXPERIMENT": True,
    "LOGGING_MODE": "online", 
    
    "TRAIN_CSV": os.path.join("data", "train_clean_weighted.csv"),
    "TRAIN_IMG_DIR": os.path.join("data", "training_data", "training_data"),
    "TEST_IMG_DIR": os.path.join("data", "test_data", "test_data"),
    "SUBMISSION_TEMPLATE": os.path.join("data", "sample_submission.csv"),
    "BAD_IMG_CSV": os.path.join("data", "bad_images.csv"),
    
    # Kaggle strategy: Train on EVERYTHING
    "USE_CLEAN_DATA": False,     
    "SNAP_SUBMISSION": "angle",  
    
    "SPEED_AS_CLASSIFICATION": True, 
    "LOSS_WEIGHT_ANGLE": 1.0,        
    "LOSS_WEIGHT_SPEED": 5.0,        
    
    # Bumped resolution for better feature extraction
    "IMG_WIDTH_TARGET": 256,  
    "IMG_HEIGHT_TARGET": 256,
    "CROP_TOP_PIXELS": 60, 
    "CROP_BOTTOM_PIXELS": 0, 
    "CHANNELS": 3,
    
    "AUG_USE_AUGMENTATION": True,
    "AUG_BRIGHTNESS_DELTA": 0.2,   
    "AUG_CONTRAST_LOWER": 0.8,     
    "AUG_CONTRAST_UPPER": 1.2,     
    "AUG_SATURATION_LOWER": 0.8,
    "AUG_SATURATION_UPPER": 1.2,
    "AUG_HUE_DELTA": 0.1,         
    "AUG_ROTATION_FACTOR": 0.05,
    "AUG_TILT_FACTOR": 0.05,
    "AUG_CUTOUT_PROB": 0.3,        
    "AUG_CUTOUT_MIN_PIX": 30,      
    "AUG_CUTOUT_MAX_PIX": 80,      
    
    # Simplified Training Dynamics
    "EPOCHS_WARMUP": 5,             
    "EPOCHS_FINETUNE": 15, 
    "LEARNING_RATE_WARMUP": 1e-3,
    "LEARNING_RATE_FINETUNE": 1e-5, 
    "BATCH_SIZE": 16, # Lowered to fit heavier models in VRAM
    "LOSS_FUNCTION": "huber",
    
    # --- HEAVYWEIGHT BASE MODEL ---
    "BASE_MODEL": "ConvNeXtBase", # Options: ConvNeXtBase, EfficientNetV2S, EfficientNetV2M
    "BASE_WEIGHTS": "imagenet",
    
    "DENSE_UNITS_1": 256,       
    "DROPOUT_RATE": 0.4, # Increased to prevent overfitting massive networks        
    "ACTIVATION_HIDDEN": "gelu", # GELU often performs better than RELU on modern architectures
    "ACTIVATION_OUTPUT": "sigmoid"
}

CONFIG["INPUT_SHAPE"] = (CONFIG["IMG_HEIGHT_TARGET"], CONFIG["IMG_WIDTH_TARGET"], CONFIG["CHANNELS"])

# --- DATA PIPELINE (Reused from your baseline) ---
random_rotation_layer = tf.keras.layers.RandomRotation(factor=CONFIG["AUG_ROTATION_FACTOR"], fill_mode='nearest')
random_translation_layer = tf.keras.layers.RandomTranslation(height_factor=CONFIG["AUG_TILT_FACTOR"], width_factor=0.0, fill_mode='nearest')

def augment_image(img):
    img = tf.image.random_brightness(img, max_delta=CONFIG["AUG_BRIGHTNESS_DELTA"])
    img = tf.image.random_contrast(img, lower=CONFIG["AUG_CONTRAST_LOWER"], upper=CONFIG["AUG_CONTRAST_UPPER"])
    img = tf.image.random_saturation(img, lower=CONFIG["AUG_SATURATION_LOWER"], upper=CONFIG["AUG_SATURATION_UPPER"])
    img = tf.image.random_hue(img, max_delta=CONFIG["AUG_HUE_DELTA"])
    if CONFIG["AUG_ROTATION_FACTOR"] > 0:
        img = tf.expand_dims(img, 0)
        img = random_rotation_layer(img, training=True) 
        img = tf.squeeze(img, 0)
    if CONFIG.get("AUG_TILT_FACTOR", 0) > 0:
        img = tf.expand_dims(img, 0) 
        img = random_translation_layer(img, training=True) 
        img = tf.squeeze(img, 0)
    return tf.clip_by_value(img, 0.0, 1.0)

def random_cutout(img, probability, min_pixels, max_pixels):
    def _apply_cutout(img):
        shape = tf.shape(img)
        H, W = shape[0], shape[1]
        h = tf.random.uniform([], minval=min_pixels, maxval=max_pixels, dtype=tf.int32)
        w = tf.random.uniform([], minval=min_pixels, maxval=max_pixels, dtype=tf.int32)
        y = tf.random.uniform([], minval=0, maxval=tf.maximum(1, H - h), dtype=tf.int32)
        x = tf.random.uniform([], minval=0, maxval=tf.maximum(1, W - w), dtype=tf.int32)
        yy, xx = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
        mask_y = tf.logical_and(yy >= y, yy < y + h)
        mask_x = tf.logical_and(xx >= x, xx < x + w)
        mask = tf.expand_dims(tf.logical_and(mask_y, mask_x), -1)
        return tf.where(mask, tf.zeros_like(img), img)
    do_cutout = tf.random.uniform([]) < probability
    return tf.cond(do_cutout, lambda: _apply_cutout(img), lambda: img)

def reshape_image(img):
    top = CONFIG.get("CROP_TOP_PIXELS", 0)
    bottom_crop = CONFIG.get("CROP_BOTTOM_PIXELS", 0)
    bottom = -bottom_crop if bottom_crop > 0 else None
    img = img[top:bottom, :, :]
    img = tf.image.resize(img, [CONFIG["IMG_HEIGHT_TARGET"], CONFIG["IMG_WIDTH_TARGET"]])
    return img

def read_and_decode_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=CONFIG["CHANNELS"])
    img = tf.cast(img, tf.float32) / 255.0
    return reshape_image(img)

def apply_augmentations(img, labels):
    if CONFIG.get("AUG_USE_AUGMENTATION", False):
        img = augment_image(img)
        flip_cond = tf.random.uniform([]) < 0.5
        img = tf.cond(flip_cond, lambda: tf.image.flip_left_right(img), lambda: img)
        new_angle = tf.cond(flip_cond, lambda: 1.0 - labels['angle_output'], lambda: labels['angle_output'])
        labels = {'angle_output': new_angle, 'speed_output': labels['speed_output']}
        img = random_cutout(img, probability=CONFIG["AUG_CUTOUT_PROB"], min_pixels=CONFIG["AUG_CUTOUT_MIN_PIX"], max_pixels=CONFIG["AUG_CUTOUT_MAX_PIX"])
    return img, labels

def prepare_data_pipelines():
    print(f"[INFO] Loading data (USE_CLEAN_DATA = {CONFIG.get('USE_CLEAN_DATA')})...")
    df = pd.read_csv(CONFIG["TRAIN_CSV"])
    
    if CONFIG.get("USE_CLEAN_DATA"):
        bad_df = pd.read_csv(CONFIG["BAD_IMG_CSV"])
        bad_list = bad_df['filename'].astype(str).tolist()
        df['check_name'] = df['image_id'].astype(float).astype(int).astype(str) + '.png'
        df = df[~df['check_name'].isin(bad_list)].copy().drop(columns=['check_name'])
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Joint-distribution weighting logic
    train_df['angle_bin'] = pd.cut(train_df['angle'], bins=10, labels=False, include_lowest=True)
    train_df['speed_bin'] = pd.cut(train_df['speed'], bins=2, labels=False, include_lowest=True)
    joint_counts = train_df.groupby(['angle_bin', 'speed_bin']).size()
    total_samples, num_bins = len(train_df), len(joint_counts)
    
    def get_weight(row):
        count = joint_counts.get((row['angle_bin'], row['speed_bin']), 1)
        return total_samples / (num_bins * count)
        
    train_df['weight'] = train_df.apply(get_weight, axis=1)
    train_df = train_df.sample(n=len(train_df), replace=True, weights='weight', random_state=42)
    
    def create_ds(dataframe, is_training):
        paths = dataframe['filepath'].values
        angles = dataframe['angle'].values.astype(np.float32)
        speeds = dataframe['speed'].values.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, angles, speeds))
        ds = ds.map(lambda path, angle, speed: (read_and_decode_image(path), {'angle_output': angle, 'speed_output': speed}), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache() 
        if is_training:
            ds = ds.shuffle(buffer_size=len(dataframe))  
            ds = ds.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE) 
        return ds.batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)
    
    return create_ds(train_df, is_training=True), create_ds(val_df, is_training=False)

# --- MODEL ARCHITECTURE ---
def build_initial_model():
    inputs = tf.keras.Input(shape=CONFIG["INPUT_SHAPE"])
    print(f"[INFO] Loading {CONFIG['BASE_MODEL']} as feature extractor...")
    
    # Universal Heavyweight Loader
    if CONFIG["BASE_MODEL"] == "ConvNeXtBase":
        base_model = ConvNeXtBase(input_shape=CONFIG["INPUT_SHAPE"], include_top=False, weights=CONFIG["BASE_WEIGHTS"])
    elif CONFIG["BASE_MODEL"] == "EfficientNetV2S":
        base_model = EfficientNetV2S(input_shape=CONFIG["INPUT_SHAPE"], include_top=False, weights=CONFIG["BASE_WEIGHTS"])
    elif CONFIG["BASE_MODEL"] == "EfficientNetV2M":
        base_model = EfficientNetV2M(input_shape=CONFIG["INPUT_SHAPE"], include_top=False, weights=CONFIG["BASE_WEIGHTS"])
    else:
        raise ValueError("Unsupported Heavyweight BASE_MODEL.")

    base_model.trainable = False 
    x = base_model(inputs, training=False) 
    
    # Generic Kaggle Head (No custom attention blocks needed for these massive models)
    x = layers.GlobalAveragePooling2D(name="global_gap")(x)
    
    # Angle Branch
    x_angle = layers.Dense(CONFIG["DENSE_UNITS_1"], activation=CONFIG["ACTIVATION_HIDDEN"])(x)
    x_angle = layers.Dropout(CONFIG["DROPOUT_RATE"])(x_angle)
    angle_out = layers.Dense(1, activation=CONFIG["ACTIVATION_OUTPUT"], name='angle_output')(x_angle)

    # Speed Branch
    x_speed = layers.Dense(CONFIG["DENSE_UNITS_1"], activation=CONFIG["ACTIVATION_HIDDEN"])(x)
    x_speed = layers.Dropout(CONFIG["DROPOUT_RATE"])(x_speed)
    speed_out = layers.Dense(1, activation=CONFIG["ACTIVATION_OUTPUT"], name='speed_output')(x_speed)

    model = models.Model(inputs=inputs, outputs=[angle_out, speed_out])
    
    opt = optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE_WARMUP"], clipnorm=1.0)
    speed_loss = "binary_crossentropy" if CONFIG.get("SPEED_AS_CLASSIFICATION", False) else CONFIG["LOSS_FUNCTION"]
    speed_metrics = ["accuracy", "mse"] if CONFIG.get("SPEED_AS_CLASSIFICATION", False) else ["mse"]
    
    model.compile(
        optimizer=opt,
        loss={'angle_output': CONFIG["LOSS_FUNCTION"], 'speed_output': speed_loss},
        loss_weights={'angle_output': CONFIG.get("LOSS_WEIGHT_ANGLE", 1.0), 'speed_output': CONFIG.get("LOSS_WEIGHT_SPEED", 1.0)},
        metrics={'angle_output': ['mse'], 'speed_output': speed_metrics}
    )
    return model, base_model

class WandbLRCustomLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if wandb.run is not None:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            wandb.log({"learning_rate": lr}, commit=False)

def generate_comprehensive_predictions(model, CONFIG, exp_dir):
    print("\n[INFO] Generating comprehensive predictions file...")
    train_base = pd.read_csv(CONFIG["TRAIN_CSV"])
    test_base = pd.read_csv(CONFIG["SUBMISSION_TEMPLATE"])
    
    train_base['split'] = 'unknown'
    t_df, v_df = train_test_split(train_base, test_size=0.2, random_state=42)
    train_base.loc[train_base['image_id'].isin(t_df['image_id']), 'split'] = 'train'
    train_base.loc[train_base['image_id'].isin(v_df['image_id']), 'split'] = 'val'
    
    results = []
    huber = tf.keras.losses.Huber()
    mse = tf.keras.losses.MeanSquaredError()
    
    def process_df(df, default_split):
        for idx, row in df.iterrows():
            if 'filepath' in row and pd.notna(row['filepath']):
                img_path = row['filepath']
            else:
                filename = str(int(float(row['image_id']))) + '.png'
                img_path = os.path.join(CONFIG["TEST_IMG_DIR"], filename)
                
            if not os.path.exists(img_path): continue
            
            img_raw = tf.io.read_file(img_path)
            img = tf.image.decode_png(img_raw, channels=3)
            img = tf.cast(img, tf.float32) / 255.0
            
            top = CONFIG.get("CROP_TOP_PIXELS", 0)
            img = img[top:, :, :]
            img = tf.image.resize(img, [CONFIG["IMG_HEIGHT_TARGET"], CONFIG["IMG_WIDTH_TARGET"]])
            img_tensor = tf.expand_dims(img, 0)
            
            start = time.time()
            preds = model(img_tensor, training=False) 
            inf_time = time.time() - start
            
            p_angle, p_speed = float(preds[0][0][0]), float(preds[1][0][0])
            s_angle = round(p_angle * 15.0) / 15.0
            s_speed = 1.0 if p_speed > 0.5 else 0.0
            
            true_a = row.get('angle', np.nan)
            true_s = row.get('speed', np.nan)
            
            results.append({
                'image_id': row['image_id'],
                'split': row.get('split', default_split),
                'inference_time_sec': inf_time,
                'true_angle': true_a, 'true_speed': true_s,
                'pred_angle': p_angle, 'pred_speed': p_speed,
                'mse_loss_angle': float(mse([true_a], [p_angle])) if not np.isnan(true_a) else np.nan,
                'mse_loss_speed': float(mse([true_s], [p_speed])) if not np.isnan(true_s) else np.nan,
            })
            
    process_df(train_base, 'train')
    process_df(test_base, 'test')
    pd.DataFrame(results).to_csv(os.path.join(exp_dir, "comprehensive_predictions.csv"), index=False)

def validate_image_paths(df, img_dir):
    df = df.dropna(subset=['image_id']) 
    valid_paths, valid_indices = [], []
    for idx, row in df.iterrows():
        filename = str(int(float(row['image_id'])))
        if not filename.endswith('.png'): filename += '.png'
        full_path = os.path.join(img_dir, filename).replace("\\", "/") 
        if os.path.exists(full_path):
            valid_paths.append(full_path)
            valid_indices.append(idx)
    return valid_paths, df.loc[valid_indices].copy()

# --- MAIN ORCHESTRATOR ---
def main():
    print(f"\n========== STARTING KAGGLE EXPERIMENT: {CONFIG['EXPERIMENT_NAME']} ==========")
    exp_dir = os.path.join("experiments", CONFIG["EXPERIMENT_NAME"])
    os.makedirs(exp_dir, exist_ok=True)
    
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=CONFIG["EXPERIMENT_NAME"], 
               dir=exp_dir, config=CONFIG, notes=CONFIG["DESCRIPTION"], mode=CONFIG["LOGGING_MODE"])
    
    train_ds, val_ds = prepare_data_pipelines()
    model, base_model = build_initial_model() 
    
    callbacks = [
        WandbMetricsLogger(),
        WandbLRCustomLogger(),  
        ModelCheckpoint(filepath=os.path.join(exp_dir, "best_model.h5"), monitor="val_loss", save_best_only=True, verbose=1),
        # Add ReduceLROnPlateau for robust fine-tuning
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]
    
    # ------------------------------------------------------------------------------------
    # PHASE 1: WARM-UP 
    # ------------------------------------------------------------------------------------
    print(f"\n[INFO] --- PHASE 1: WARM-UP ({CONFIG['EPOCHS_WARMUP']} Epochs) ---")
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_WARMUP"], callbacks=callbacks)
    
    # ------------------------------------------------------------------------------------
    # PHASE 2: UNIVERSAL FINE-TUNING (Unfreeze All, Lock BatchNorm)
    # ------------------------------------------------------------------------------------
    print(f"\n[INFO] --- PHASE 2: UNIVERSAL FINE-TUNING ---")
    base_model.trainable = True 
    
    # Universal backward lock for any architecture
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False   
            
    opt = optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE_FINETUNE"], clipnorm=1.0)
    speed_loss = "binary_crossentropy" if CONFIG.get("SPEED_AS_CLASSIFICATION", False) else CONFIG["LOSS_FUNCTION"]
    speed_metrics = ["accuracy", "mse"] if CONFIG.get("SPEED_AS_CLASSIFICATION", False) else ["mse"]
    
    model.compile(
        optimizer=opt,
        loss={'angle_output': CONFIG["LOSS_FUNCTION"], 'speed_output': speed_loss},
        loss_weights={'angle_output': CONFIG.get("LOSS_WEIGHT_ANGLE", 1.0), 'speed_output': CONFIG.get("LOSS_WEIGHT_SPEED", 1.0)},
        metrics={'angle_output': ['mse'], 'speed_output': speed_metrics}
    )
    
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_WARMUP"] + CONFIG["EPOCHS_FINETUNE"], 
              initial_epoch=CONFIG["EPOCHS_WARMUP"], callbacks=callbacks)

    # ------------------------------------------------------------------------------------
    # SUBMISSION GENERATOR
    # ------------------------------------------------------------------------------------
    print("\n[INFO] Generating Kaggle submission...")
    best_model = tf.keras.models.load_model(os.path.join(exp_dir, "best_model.h5"))
    sub_df = pd.read_csv(CONFIG["SUBMISSION_TEMPLATE"])
    
    test_paths, sub_df = validate_image_paths(sub_df, CONFIG["TEST_IMG_DIR"])
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(read_and_decode_image, num_parallel_calls=tf.data.AUTOTUNE).batch(CONFIG["BATCH_SIZE"])
    
    predictions = best_model.predict(test_ds)
    submission_df = pd.DataFrame({'image_id': sub_df['image_id'], 'angle': predictions[0].flatten(), 'speed': predictions[1].flatten()})
    
    if CONFIG.get("SNAP_SUBMISSION") in ["angle", "both"]:
        submission_df['angle'] = np.round(submission_df['angle'] * 15.0) / 15.0
    if CONFIG.get("SNAP_SUBMISSION") in ["speed", "both"]:
        submission_df['speed'] = np.where(submission_df['speed'] > 0.5, 1.0, 0.0)
        
    submission_df.to_csv(os.path.join(exp_dir, "submission.csv"), index=False)

    generate_comprehensive_predictions(best_model, CONFIG, exp_dir)    
    if wandb.run is not None: wandb.finish()
    
if __name__ == "__main__":
    main()