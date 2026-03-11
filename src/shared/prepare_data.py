"""
Shared Data Preparation 
Validates images, computes inverse-frequency weights for data balancing.
Original concept by prepare_dataset.py, modularised for shared use.

Usage: python src/shared/prepare_data.py
"""

import os
import pandas as pd
import numpy as np


def find_image_dir(base_path='data'):
    """Find the training image directory (handles double-nested folders)."""
    candidates = [
        os.path.join(base_path, 'training_data', 'training_data'),
        os.path.join(base_path, 'training_data'),
    ]
    for path in candidates:
        if os.path.exists(path) and any(f.endswith('.png') for f in os.listdir(path)):
            return path
    raise FileNotFoundError(f"Cannot find training images in {candidates}")


def find_test_dir(base_path='data'):
    """Find the test image directory (handles double-nested folders)."""
    candidates = [
        os.path.join(base_path, 'test_data', 'test_data'),
        os.path.join(base_path, 'test_data'),
    ]
    for path in candidates:
        if os.path.exists(path) and any(f.endswith('.png') for f in os.listdir(path)):
            return path
    raise FileNotFoundError(f"Cannot find test images in {candidates}")


def prepare_weighted_data(data_dir='data'):
    """
    Load train.csv, validate images exist, compute inverse-frequency weights.

    Returns:
        DataFrame with columns: image_id, angle, speed, filepath, sample_weight
    """
    csv_path = os.path.join(data_dir, 'train.csv')
    img_dir = find_image_dir(data_dir)

    print("[INFO] Loading training data...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['image_id'])

    # Validate image files exist and are non-empty
    print("[INFO] Validating image files...")
    valid_rows = []
    for idx, row in df.iterrows():
        filename = f"{int(row['image_id'])}.png"
        full_path = os.path.join(img_dir, filename)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            valid_rows.append({
                'image_id': int(row['image_id']),
                'angle': row['angle'],
                'speed': row['speed'],
                'filepath': full_path,
            })

    clean_df = pd.DataFrame(valid_rows)
    print(f"[INFO] Valid images: {len(clean_df)} / {len(df)}")

    # Compute inverse-frequency weights for data balancing
    # Rare (angle, speed) combos get higher weight
    clean_df['angle_bin'] = clean_df['angle'].round(3)
    clean_df['speed_bin'] = clean_df['speed'].round(3)
    clean_df['pair_freq'] = clean_df.groupby(['angle_bin', 'speed_bin'])['angle_bin'].transform('count')
    clean_df['sample_weight'] = 1.0 / clean_df['pair_freq']

    # Print balance summary
    stopped = (clean_df['speed'] == 0).sum()
    print(f"[INFO] Stopped: {stopped} ({100*stopped/len(clean_df):.1f}%), Moving: {len(clean_df)-stopped}")
    print(f"[INFO] Unique angle values: {clean_df['angle'].nunique()}")

    return clean_df


if __name__ == '__main__':
    df = prepare_weighted_data()
    print(f"\nDataset ready: {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"\nWeight stats:")
    print(df['sample_weight'].describe())
