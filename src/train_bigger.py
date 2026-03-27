import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ConvNeXtBase, EfficientNetV2S, EfficientNetV2M, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import wandb
from wandb.integration.keras import WandbMetricsLogger

# ==============================================================================
# GPU SETUP & MIXED PRECISION (Keep Disabled for CPU Local Inference)
# ==============================================================================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] VRAM memory growth enabled. GPU is ready!\n")
        print("[INFO] Enabling Mixed Precision (Float16) for GPU...")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    except RuntimeError as e:
        print(f"[ERROR] GPU Setup failed: {e}\n")
else:
    print("[WARNING] No GPU detected. Running in standard Float32 mode for CPU.")

# ==============================================================================
# 1. KAGGLE CONFIGURATION 
# ==============================================================================
WANDB_PROJECT = "PiCar"
WANDB_ENTITY = "lpxdv2-university-of-nottingham"  

CONFIG = {
    "EXPERIMENT_NAME": "38_V2S_crop_lr_speed",
    "DESCRIPTION": "Crop bottom, raise learning rate, keep early layers frozen",
    "OVERWRITE_EXPERIMENT": True,
    "LOGGING_MODE": "online", 
    
    "TRAINING_MODE": "speed",  # Options: "angle" or "speed" or "both"
    
    "TRAIN_CSV": os.path.join("data", "train_clean_weighted.csv"),
    "TRAIN_IMG_DIR": os.path.join("data", "training_data", "training_data"),
    "TEST_IMG_DIR": os.path.join("data", "test_data", "test_data"),
    "SUBMISSION_TEMPLATE": os.path.join("data", "sample_submission.csv"),
    "BAD_IMG_CSV": os.path.join("data", "bad_images.csv"),
    
    "USE_CLEAN_DATA": False,     
    "SNAP_SUBMISSION": None,  
    
    "SPEED_AS_CLASSIFICATION": True, 
    "LOSS_FUNCTION": "huber",
    
    "IMG_WIDTH_TARGET": 224,  
    "IMG_HEIGHT_TARGET": 224,
    "CROP_TOP_PIXELS": 60, 
    "CROP_BOTTOM_PIXELS": 40, 
    "CHANNELS": 3,
    
    "AUG_USE_AUGMENTATION": True,
    "AUG_BRIGHTNESS_DELTA": 0.2,   
    "AUG_CONTRAST_LOWER": 0.8,     
    "AUG_CONTRAST_UPPER": 1.2,     
    "AUG_SATURATION_LOWER": 0.8,
    "AUG_SATURATION_UPPER": 1.2,
    "AUG_HUE_DELTA": 0.1,         
    "AUG_ROTATION_FACTOR": 0.005,
    "AUG_TILT_FACTOR": 0.005,
    "AUG_CUTOUT_PROB": 0.3,        
    "AUG_CUTOUT_MIN_PIX": 30,      
    "AUG_CUTOUT_MAX_PIX": 80,      
    
    "EPOCHS_WARMUP": 5,             
    "EPOCHS_FINETUNE": 15, 
    "LEARNING_RATE_WARMUP": 2e-3,
    "LEARNING_RATE_FINETUNE": 3e-5, 
    "BATCH_SIZE": 16, 
    
    "BASE_MODEL": "EfficientNetV2S",
    "BASE_WEIGHTS": "imagenet",
    
    # --- Progressive Unfreezing Hyperparameters ---
    "START_UNFREEZE_BLOCK": 6,      # EfficientNetV2S has 6 main blocks. Start at the top.
    "FREEZE_UP_TO_BLOCK": 3,        # Unfreeze down to block 4. (Blocks 1, 2, 3, and stem stay frozen permanently)
    "EPOCHS_PER_UNFREEZE_STEP": 6,  # Train for 4 epochs every time we wake up a new block
    "UNFREEZE_LR_DECAY": 0.9,       # Drop LR by 20% each time we go deeper
    
    "DENSE_UNITS_1": 256, 
    "DENSE_UNITS_2": 64,       
    "DROPOUT_RATE": 0.2, 
    "ACTIVATION_HIDDEN": "gelu", 
    "ACTIVATION_OUTPUT": "sigmoid"
}

CONFIG["INPUT_SHAPE"] = (CONFIG["IMG_HEIGHT_TARGET"], CONFIG["IMG_WIDTH_TARGET"], CONFIG["CHANNELS"])

# --- DATA PIPELINE ---
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
        
        # --- GRAPH-SAFE MIRRORED HORIZONTAL FLIP ---
        flip_cond = tf.random.uniform([]) < 0.5
        img = tf.cond(flip_cond, lambda: tf.image.flip_left_right(img), lambda: img)
        
        # Only invert the angle. Speed stays the same.
        new_angle = tf.cond(flip_cond, lambda: 1.0 - labels['angle_output'], lambda: labels['angle_output'])
        new_speed = labels['speed_output']
            
        labels = {'angle_output': new_angle, 'speed_output': new_speed}
        img = random_cutout(img, probability=CONFIG["AUG_CUTOUT_PROB"], min_pixels=CONFIG["AUG_CUTOUT_MIN_PIX"], max_pixels=CONFIG["AUG_CUTOUT_MAX_PIX"])
    return img, labels

def prepare_data_pipelines():
    print(f"[INFO] Loading data for target: {CONFIG['TRAINING_MODE'].upper()}")
    df = pd.read_csv(CONFIG["TRAIN_CSV"])
    
    if CONFIG.get("USE_CLEAN_DATA"):
        bad_df = pd.read_csv(CONFIG["BAD_IMG_CSV"])
        bad_list = bad_df['filename'].astype(str).tolist()
        df['check_name'] = df['image_id'].astype(float).astype(int).astype(str) + '.png'
        df = df[~df['check_name'].isin(bad_list)].copy().drop(columns=['check_name'])
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
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
        # Extract BOTH targets
        angle_targets = dataframe['angle'].values.astype(np.float32)
        speed_targets = dataframe['speed'].values.astype(np.float32)
        
        ds = tf.data.Dataset.from_tensor_slices((paths, angle_targets, speed_targets))
        # Map them explicitly to the two hardcoded layer names
        ds = ds.map(lambda path, ang, spd: (
            read_and_decode_image(path), 
            {'angle_output': ang, 'speed_output': spd}
        ), num_parallel_calls=tf.data.AUTOTUNE)
        
        ds = ds.cache() 
        if is_training:
            ds = ds.shuffle(buffer_size=len(dataframe))  
            ds = ds.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE) 
        return ds.batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)
    
    return create_ds(train_df, is_training=True), create_ds(val_df, is_training=False)

# --- MODEL COMPILE HELPER ---
def get_compile_args(lr):
    opt = optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    
    # 1. Routing: Turn heads on or off
    if CONFIG["TRAINING_MODE"] == "angle":
        weights = {'angle_output': 1.0, 'speed_output': 0.0}
    elif CONFIG["TRAINING_MODE"] == "speed":
        weights = {'angle_output': 0.0, 'speed_output': 1.0}
    else: # "both"
        weights = {'angle_output': 1.0, 'speed_output': 1.0}

    # 2. Losses
    speed_loss = "binary_crossentropy" if CONFIG.get("SPEED_AS_CLASSIFICATION") else CONFIG["LOSS_FUNCTION"]
    losses = {'angle_output': CONFIG["LOSS_FUNCTION"], 'speed_output': speed_loss}
    
    # 3. Metrics (MSE is locked in for both!)
    speed_metrics = ["accuracy", "mse"] if CONFIG.get("SPEED_AS_CLASSIFICATION") else ["mse"]
    metrics = {'angle_output': ['mse'], 'speed_output': speed_metrics}

    return {"optimizer": opt, "loss": losses, "loss_weights": weights, "metrics": metrics}

# --- MODEL ARCHITECTURE ---
def build_initial_model():
    inputs = tf.keras.Input(shape=CONFIG["INPUT_SHAPE"])
    print(f"[INFO] Loading {CONFIG['BASE_MODEL']} as feature extractor...")
    
    if CONFIG["BASE_MODEL"] == "ConvNeXtBase":
        base_model = ConvNeXtBase(input_shape=CONFIG["INPUT_SHAPE"], include_top=False, weights=CONFIG["BASE_WEIGHTS"])
    elif CONFIG["BASE_MODEL"] == "EfficientNetV2S":
        base_model = EfficientNetV2S(input_shape=CONFIG["INPUT_SHAPE"], include_top=False, weights=CONFIG["BASE_WEIGHTS"])
    elif CONFIG["BASE_MODEL"] == "EfficientNetV2M":
        base_model = EfficientNetV2M(input_shape=CONFIG["INPUT_SHAPE"], include_top=False, weights=CONFIG["BASE_WEIGHTS"])
    elif CONFIG["BASE_MODEL"] == "EfficientNetB0":
        base_model = EfficientNetB0(input_shape=CONFIG["INPUT_SHAPE"], include_top=False, weights=CONFIG["BASE_WEIGHTS"])
    else:
        raise ValueError("Unsupported BASE_MODEL.")

    base_model.trainable = False  # <--- MUST ADD THIS BACK
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_gap")(x)
    
    # --- ANGLE BRANCH ---
    a = layers.Dense(CONFIG["DENSE_UNITS_1"], activation=CONFIG["ACTIVATION_HIDDEN"])(x)
    a = layers.Dropout(CONFIG["DROPOUT_RATE"])(a)
    angle_output = layers.Dense(1, activation=CONFIG["ACTIVATION_OUTPUT"], name="angle_output")(a)

    # --- SPEED BRANCH ---
    s = layers.Dense(CONFIG["DENSE_UNITS_1"], activation=CONFIG["ACTIVATION_HIDDEN"])(x)
    s = layers.Dropout(CONFIG["DROPOUT_RATE"])(s)
    speed_output = layers.Dense(1, activation=CONFIG["ACTIVATION_OUTPUT"], name="speed_output")(s)

    model = models.Model(inputs=inputs, outputs=[angle_output, speed_output])
    model.compile(**get_compile_args(CONFIG["LEARNING_RATE_WARMUP"]))
    return model, base_model

class WandbLRCustomLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if wandb.run is not None:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            wandb.log({"learning_rate": lr}, commit=False)

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
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]
    
    print(f"\n[INFO] --- PHASE 1: WARM-UP ({CONFIG['EPOCHS_WARMUP']} Epochs) ---")
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS_WARMUP"], callbacks=callbacks)
    
# ------------------------------------------------------------------------------------
    # PHASE 2: PROGRESSIVE UNFREEZING (EfficientNetV2S Blocks)
    # ------------------------------------------------------------------------------------
    start_block = CONFIG["START_UNFREEZE_BLOCK"]
    end_block = CONFIG["FREEZE_UP_TO_BLOCK"]
    
    current_lr = CONFIG["LEARNING_RATE_FINETUNE"]
    current_epoch = CONFIG["EPOCHS_WARMUP"]
    epochs_per_step = CONFIG["EPOCHS_PER_UNFREEZE_STEP"]
    lr_decay = CONFIG["UNFREEZE_LR_DECAY"]

    base_model.trainable = True 
    
    # 1. ALWAYS lock BatchNormalization layers to prevent moving-variance explosions
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False   

    # 2. Iterate backwards from Block 6 down to your target block
    for target_block in range(start_block, end_block - 1, -1):
        print(f"\n[INFO] --- PHASE 2: UNFREEZING DOWN TO BLOCK {target_block} ---")
        print(f"[INFO] Current Learning Rate: {current_lr:.2e}")
        
        # In EfficientNetV2S, blocks are named "block1a_", "block2b_", etc.
        # We freeze any block NUMBER less than our target_block.
        blocks_to_freeze = [f"block{i}" for i in range(1, target_block)]
        
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                continue # Already locked above
                
            is_stem = layer.name.startswith("stem")
            
            # If the layer is the stem, or starts with one of our frozen block names, lock it.
            if is_stem or any(layer.name.startswith(b) for b in blocks_to_freeze):
                layer.trainable = False
            else:
                layer.trainable = True

        # 3. We MUST recompile the model every time we change trainable variables
        model.compile(**get_compile_args(current_lr))
        
        target_epoch = current_epoch + epochs_per_step
        
        model.fit(train_ds, validation_data=val_ds, 
                  epochs=target_epoch, 
                  initial_epoch=current_epoch, 
                  callbacks=callbacks)
        
        current_epoch = target_epoch
        current_lr *= lr_decay # Soften the learning rate as we go deeper

    # ------------------------------------------------------------------------------------
    # DYNAMIC BLANK-COLUMN SUBMISSION GENERATOR
    # ------------------------------------------------------------------------------------
    print("\n[INFO] Generating Kaggle submission...")
    best_model = tf.keras.models.load_model(os.path.join(exp_dir, "best_model.h5"))
    sub_df = pd.read_csv(CONFIG["SUBMISSION_TEMPLATE"])
    
    test_paths, sub_df = validate_image_paths(sub_df, CONFIG["TEST_IMG_DIR"])
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(read_and_decode_image, num_parallel_calls=tf.data.AUTOTUNE).batch(CONFIG["BATCH_SIZE"])
    
    # The model now returns a list of two arrays
    preds = best_model.predict(test_ds)
    angle_preds = preds[0].flatten()
    speed_preds = preds[1].flatten()
    
    if CONFIG.get("SNAP_SUBMISSION") in ["angle", "both"]:
        angle_preds = np.round(angle_preds * 15.0) / 15.0
    if CONFIG.get("SNAP_SUBMISSION") in ["speed", "both"]:
        speed_preds = np.where(speed_preds > 0.5, 1.0, 0.0)
        
    submission_df = pd.DataFrame({
        'image_id': sub_df['image_id'], 
        'angle': angle_preds, 
        'speed': speed_preds
    })
    
    submission_path = os.path.join(exp_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"[SUCCESS] Partial Submission saved to: {submission_path}")

    if wandb.run is not None: wandb.finish()
    
if __name__ == "__main__":
    main()