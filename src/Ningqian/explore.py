"""
Data Exploration

Usage: python src/explore.py
"""

import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for remote server
import matplotlib.pyplot as plt


def main():
    # ============================================================
    # 1. Load and inspect train.csv
    # ============================================================
    print("=" * 60)
    print("1. LOADING train.csv")
    print("=" * 60)

    df = pd.read_csv('data/train.csv')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nBasic statistics:")
    print(df.describe())

    # ============================================================
    # 2. Check value ranges (already normalised?)
    # ============================================================
    print("\n" + "=" * 60)
    print("2. VALUE RANGES")
    print("=" * 60)
    print(f"Angle: min={df['angle'].min():.4f}, max={df['angle'].max():.4f}, mean={df['angle'].mean():.4f}")
    print(f"Speed: min={df['speed'].min():.4f}, max={df['speed'].max():.4f}, mean={df['speed'].mean():.4f}")
    print(f"\nUnique speed values: {sorted(df['speed'].unique())}")
    print(f"Number of unique angle values: {df['angle'].nunique()}")

    # Check how many stopped (speed=0) vs moving
    stopped = (df['speed'] == 0).sum()
    moving = (df['speed'] > 0).sum()
    print(f"\nStopped (speed=0): {stopped} ({100*stopped/len(df):.1f}%)")
    print(f"Moving (speed>0):  {moving} ({100*moving/len(df):.1f}%)")

    # ============================================================
    # 3. Visualise distributions
    # ============================================================
    print("\n" + "=" * 60)
    print("3. SAVING DISTRIBUTION PLOTS")
    print("=" * 60)

    os.makedirs('plots', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Angle distribution
    axes[0, 0].hist(df['angle'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Steering Angle Distribution')
    axes[0, 0].set_xlabel('Normalised Angle')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Center (0.5)')
    axes[0, 0].legend()

    # Speed distribution
    axes[0, 1].hist(df['speed'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_title('Speed Distribution')
    axes[0, 1].set_xlabel('Normalised Speed')
    axes[0, 1].set_ylabel('Count')

    # Angle vs Speed scatter
    axes[1, 0].scatter(df['angle'], df['speed'], alpha=0.1, s=1)
    axes[1, 0].set_title('Angle vs Speed')
    axes[1, 0].set_xlabel('Normalised Angle')
    axes[1, 0].set_ylabel('Normalised Speed')

    # Angle distribution for moving cars only
    moving_df = df[df['speed'] > 0]
    axes[1, 1].hist(moving_df['angle'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].set_title('Steering Angle (Moving Cars Only)')
    axes[1, 1].set_xlabel('Normalised Angle')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].axvline(x=0.5, color='red', linestyle='--', label='Center (0.5)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('plots/distributions.png', dpi=150)
    print("Saved: plots/distributions.png")

    # ============================================================
    # 4. Check image dimensions and view samples
    # ============================================================
    print("\n" + "=" * 60)
    print("4. IMAGE INSPECTION")
    print("=" * 60)

    # Try both possible paths (depends on how unzip was done)
    if os.path.exists('data/training_data/training_data'):
        img_dir = 'data/training_data/training_data'
    elif os.path.exists('data/training_data'):
        img_dir = 'data/training_data'
    else:
        print("ERROR: Cannot find training images!")
        print("Checked: data/training_data/training_data and data/training_data")
        return

    print(f"Image directory: {img_dir}")

    # Check first image
    first_img_path = os.path.join(img_dir, '0.png')
    if os.path.exists(first_img_path):
        img = Image.open(first_img_path)
        print(f"Image size: {img.size} (width x height)")
        print(f"Image mode: {img.mode}")
    else:
        # Find any image
        imgs = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        print(f"Found {len(imgs)} images")
        if imgs:
            img = Image.open(os.path.join(img_dir, imgs[0]))
            print(f"Image size: {img.size} (width x height)")
            print(f"Image mode: {img.mode}")

    # Count total images
    total_imgs = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
    print(f"Total training images: {total_imgs}")

    # ============================================================
    # 5. View sample images at different angles/speeds
    # ============================================================
    print("\n" + "=" * 60)
    print("5. SAVING SAMPLE IMAGES")
    print("=" * 60)

    # Sample images: hard left, center, hard right, stopped
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Top row: different steering angles
    angle_bins = [
        ('Hard Left', df[df['angle'] < 0.2]),
        ('Slight Left', df[(df['angle'] >= 0.3) & (df['angle'] < 0.45)]),
        ('Center', df[(df['angle'] >= 0.45) & (df['angle'] <= 0.55)]),
        ('Hard Right', df[df['angle'] > 0.8]),
    ]

    for i, (label, subset) in enumerate(angle_bins):
        if len(subset) > 0:
            row = subset.sample(1).iloc[0]
            img_path = os.path.join(img_dir, f"{int(row['image_id'])}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"{label}\nangle={row['angle']:.3f}, speed={row['speed']:.1f}")
            else:
                axes[0, i].set_title(f"{label}\n(image not found)")
        else:
            axes[0, i].set_title(f"{label}\n(no samples)")
        axes[0, i].axis('off')

    # Bottom row: stopped vs moving, and edge cases
    speed_bins = [
        ('Stopped', df[df['speed'] == 0]),
        ('Moving (speed=1)', df[df['speed'] == 1.0]),
        ('Low Speed', df[(df['speed'] > 0) & (df['speed'] < 0.5)]),
        ('Random Sample', df),
    ]

    for i, (label, subset) in enumerate(speed_bins):
        if len(subset) > 0:
            row = subset.sample(1).iloc[0]
            img_path = os.path.join(img_dir, f"{int(row['image_id'])}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[1, i].imshow(img)
                axes[1, i].set_title(f"{label}\nangle={row['angle']:.3f}, speed={row['speed']:.1f}")
            else:
                axes[1, i].set_title(f"{label}\n(image not found)")
        else:
            axes[1, i].set_title(f"{label}\n(no samples)")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('plots/sample_images.png', dpi=150)
    print("Saved: plots/sample_images.png")

    # ============================================================
    # 6. Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("6. SUMMARY")
    print("=" * 60)
    print(f"Training samples: {len(df)}")
    print(f"Training images:  {total_imgs}")
    print(f"Test images:      (check data/test_data/)")
    print(f"Angle range:      [{df['angle'].min():.4f}, {df['angle'].max():.4f}]")
    print(f"Speed values:     {sorted(df['speed'].unique())}")
    print(f"Stopped samples:  {stopped} ({100*stopped/len(df):.1f}%)")

if __name__ == '__main__':
    main()
