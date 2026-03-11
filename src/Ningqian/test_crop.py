"""
Visual test: compare cropping and resize options on 5 random images.

Shows: Original → Cropped → Proportional Resize → NVIDIA Resize

Usage: python src/test_crop.py
"""

import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Config
CROP_TOP = 80       # Remove sky/ceiling
CROP_BOTTOM = 20    # Remove car hood
RESIZE_WIDTH = 200  # Target width

# Find images
img_dir = os.path.join('data', 'training_data', 'training_data')
all_images = [f for f in os.listdir(img_dir) if f.endswith('.png') and os.path.getsize(os.path.join(img_dir, f)) > 0]
random.seed(42)
sample_images = random.sample(all_images, 5)

print(f"Testing crop: top {CROP_TOP}px, bottom {CROP_BOTTOM}px")
print(f"Sample images: {sample_images}\n")

fig, axes = plt.subplots(5, 4, figsize=(16, 16))
col_titles = ['Original', 'After Crop', 'Proportional Resize', 'NVIDIA Resize (200x66)']

for row, fname in enumerate(sample_images):
    img_path = os.path.join(img_dir, fname)
    img = Image.open(img_path)
    w, h = img.size

    # Crop
    cropped = img.crop((0, CROP_TOP, w, h - CROP_BOTTOM))
    cw, ch = cropped.size

    # Proportional resize (keep aspect ratio)
    prop_h = int(RESIZE_WIDTH * ch / cw)
    proportional = cropped.resize((RESIZE_WIDTH, prop_h))

    # NVIDIA resize (fixed 200x66, slightly squished)
    nvidia = cropped.resize((200, 66))

    # Plot
    axes[row][0].imshow(img)
    axes[row][0].axhline(y=CROP_TOP, color='red', linestyle='--', linewidth=1.5)
    axes[row][0].axhline(y=h - CROP_BOTTOM, color='blue', linestyle='--', linewidth=1.5)
    axes[row][0].set_ylabel(fname, fontsize=9)
    if row == 0:
        axes[row][0].set_title(f'{col_titles[0]}\n({w}x{h})', fontsize=10)

    axes[row][1].imshow(cropped)
    if row == 0:
        axes[row][1].set_title(f'{col_titles[1]}\n({cw}x{ch})', fontsize=10)

    axes[row][2].imshow(proportional)
    if row == 0:
        axes[row][2].set_title(f'{col_titles[2]}\n({RESIZE_WIDTH}x{prop_h})', fontsize=10)

    axes[row][3].imshow(nvidia)
    if row == 0:
        axes[row][3].set_title(f'{col_titles[3]}\n(200x66)', fontsize=10)

    # Clean up axes
    for col in range(4):
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])

    if row == 0:
        print(f"Original: {w}x{h}")
        print(f"After crop (top {CROP_TOP}, bottom {CROP_BOTTOM}): {cw}x{ch}")
        print(f"Proportional resize: {RESIZE_WIDTH}x{prop_h} (aspect ratio kept)")
        print(f"NVIDIA resize: 200x66 (slightly squished)")
        print(f"Aspect ratio — original crop: {cw/ch:.2f}, proportional: {RESIZE_WIDTH/prop_h:.2f}, NVIDIA: {200/66:.2f}")

# Add legend for crop lines
fig.text(0.13, 0.01, '--- red = top crop line    --- blue = bottom crop line',
         fontsize=10, color='gray', ha='left')

plt.suptitle(f'Crop Test: top {CROP_TOP}px + bottom {CROP_BOTTOM}px', fontsize=14, y=1.0)
plt.tight_layout()

os.makedirs('plots', exist_ok=True)
save_path = 'plots/crop_test.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved to: {save_path}")
