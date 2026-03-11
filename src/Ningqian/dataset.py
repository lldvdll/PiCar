"""
Key features:
- Image cropping (top 80px sky + bottom 20px hood) and proportional resize to 200x87
- Data augmentation (flip, brightness, contrast, shadow)
- Contiguous block validation split (avoids data leakage)
- Inverse-frequency data balancing (from David's prepare_dataset.py)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import pandas as pd
import numpy as np
from shared.prepare_data import prepare_weighted_data, find_test_dir


# ============================================================
# Configuration
# ============================================================
IMG_HEIGHT = 87     # Proportional resize (keeps aspect ratio after crop)
IMG_WIDTH = 200
CROP_TOP = 80       # Remove sky/ceiling
CROP_BOTTOM = 20    # Remove car hood


def load_and_preprocess_image(image_path):
    """Load a PNG image, crop, resize, normalise."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)

    # Crop top (sky) and bottom (car hood)
    shape = tf.shape(img)
    height = shape[0]
    img = img[CROP_TOP:height - CROP_BOTTOM, :, :]

    # Resize to NVIDIA dimensions
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    # Normalise to [0, 1]
    img = img / 255.0
    return img


def augment_image(img, angle, speed):
    """Apply data augmentation."""
    # Random brightness
    img = tf.image.random_brightness(img, max_delta=0.2)

    # Random contrast
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)

    # Random horizontal flip (mirror steering angle around 0.5)
    if tf.random.uniform([]) > 0.5:
        img = tf.image.flip_left_right(img)
        angle = 1.0 - angle

    # Random shadow
    if tf.random.uniform([]) > 0.5:
        shadow_width = tf.random.uniform([], minval=0.2, maxval=0.5)
        shadow_start = tf.random.uniform([], minval=0.0, maxval=1.0 - shadow_width)
        shadow_end = shadow_start + shadow_width

        w = tf.cast(tf.shape(img)[1], tf.float32)
        start_px = tf.cast(shadow_start * w, tf.int32)
        end_px = tf.cast(shadow_end * w, tf.int32)

        indices = tf.range(tf.shape(img)[1])
        mask = tf.logical_and(indices >= start_px, indices < end_px)
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, [1, -1, 1])
        shadow = 1.0 - 0.5 * mask
        img = img * shadow

    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, angle, speed


def create_datasets(data_dir='data', batch_size=32, val_split=0.2, augment=True, balance_data=True):
    """
    Create training and validation datasets.

    Args:
        balance_data: If True, oversample rare (angle, speed) combinations
                      using inverse-frequency weighting (David's approach)
    """
    # Load and validate data with weights
    df = prepare_weighted_data(data_dir)

    # Contiguous block split (avoids data leakage from consecutive frames)
    df = df.sort_values('image_id').reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_split))

    train_df = df[:split_idx]
    val_df = df[split_idx:]

    # Apply data balancing to training set only
    if balance_data:
        print("[INFO] Applying inverse-frequency balancing to training data...")
        train_df = train_df.sample(
            n=len(train_df), replace=True,
            weights='sample_weight', random_state=42
        ).reset_index(drop=True)
        print(f"[INFO] Balanced training set: {len(train_df)} samples")
    else:
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Training: {len(train_df)}, Validation: {len(val_df)}")

    # Build TF datasets
    def make_generator(dataframe):
        def gen():
            for _, row in dataframe.iterrows():
                yield row['filepath'], row['angle'], row['speed']
        return gen

    output_sig = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    # Training dataset
    train_ds = tf.data.Dataset.from_generator(make_generator(train_df), output_signature=output_sig)
    train_ds = train_ds.map(
        lambda path, angle, speed: (load_and_preprocess_image(path), angle, speed),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if augment:
        train_ds = train_ds.map(
            lambda img, angle, speed: augment_image(img, angle, speed),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    train_ds = train_ds.map(
        lambda img, angle, speed: (img, tf.stack([angle, speed])),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation dataset (no augmentation, no balancing)
    val_ds = tf.data.Dataset.from_generator(make_generator(val_df), output_signature=output_sig)
    val_ds = val_ds.map(
        lambda path, angle, speed: (load_and_preprocess_image(path), angle, speed),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda img, angle, speed: (img, tf.stack([angle, speed])),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, len(train_df), len(val_df)


def create_test_dataset(data_dir='data', batch_size=32):
    """Create test dataset for generating Kaggle submission predictions."""
    test_dir = find_test_dir(data_dir)

    test_files = sorted(
        [f for f in os.listdir(test_dir) if f.endswith('.png')],
        key=lambda x: int(x.replace('.png', ''))
    )

    image_ids = [int(f.replace('.png', '')) for f in test_files]
    paths = [os.path.join(test_dir, f) for f in test_files]

    print(f"Found {len(paths)} test images")

    test_ds = tf.data.Dataset.from_tensor_slices(paths)
    test_ds = test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return test_ds, image_ids
