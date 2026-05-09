"""
data_prep.py — 2x2 grid of input / preprocessed / 2 augmentations
for report figure generation.

Top row:    raw input | cropped + resized
Bottom row: augmentation A (cutout forced) | augmentation B (cutout forced)
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "picar_autopilot_models",
                                # "57_kaggle_16_keep12_unfreeze4"))
                                "58_57_more_lighting_aug"))
from train import CONFIG, read_and_decode_image, augment_image, random_cutout


def augment_with_cutout(tensor):
    """Apply augment_image then force a cutout (probability=1.0)."""
    aug = augment_image(tensor)
    aug = random_cutout(aug,
                        probability=1.0,
                        min_pixels=CONFIG["AUG_CUTOUT_MIN_PIX"],
                        max_pixels=CONFIG["AUG_CUTOUT_MAX_PIX"])
    return aug


def load_index(csv_path, img_dir):
    print(f"[INFO] Loading index from {csv_path}...")
    df = pd.read_csv(csv_path)
    items = {}
    for _, row in df.dropna(subset=['image_id']).iterrows():
        img_id = str(int(float(row['image_id'])))
        full_path = os.path.join(img_dir, img_id + '.png').replace("\\", "/")
        if os.path.exists(full_path):
            items[img_id] = {
                'filepath': full_path,
                'angle': row.get('angle', 'N/A'),
                'speed': row.get('speed', 'N/A'),
            }
    print(f"[INFO] Indexed {len(items)} images.")
    return items


def main():
    images = load_index(CONFIG["TRAIN_CSV"], CONFIG["TRAIN_IMG_DIR"])
    if not images:
        raise ValueError("No images found! Check CONFIG paths.")
    image_ids = list(images.keys())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plt.subplots_adjust(bottom=0.14, left=0.02, right=0.98, top=0.94, hspace=0.18, wspace=0.05)

    state = {
        'current_id': random.choice(image_ids),
        'raw': None,
        'preprocessed': None,
        'aug_a': None,
        'aug_b': None,
    }

    def render(load_new=True):
        item = images[state['current_id']]

        if load_new:
            raw = tf.io.read_file(item['filepath'])
            raw = tf.image.decode_png(raw, channels=CONFIG["CHANNELS"])
            state['raw'] = raw
            state['preprocessed'] = read_and_decode_image(item['filepath'])

        state['aug_a'] = augment_with_cutout(state['preprocessed'])
        state['aug_b'] = augment_with_cutout(state['preprocessed'])

        axes[0][0].clear(); axes[0][0].axis('off')
        axes[0][0].imshow(state['raw'].numpy())
        axes[0][0].set_title(f"Raw {tuple(state['raw'].shape)}")

        axes[0][1].clear(); axes[0][1].axis('off')
        axes[0][1].imshow(state['preprocessed'].numpy())
        axes[0][1].set_title(f"Cropped + resized {tuple(state['preprocessed'].shape)}")

        axes[1][0].clear(); axes[1][0].axis('off')
        axes[1][0].imshow(state['aug_a'].numpy())
        axes[1][0].set_title("Augmentation A")

        axes[1][1].clear(); axes[1][1].axis('off')
        axes[1][1].imshow(state['aug_b'].numpy())
        axes[1][1].set_title("Augmentation B")

        fig.canvas.draw_idle()

    def cycle(_=None):
        state['current_id'] = random.choice(image_ids)
        text_box.set_val(state['current_id'])
        render(load_new=True)

    def jump_to(text):
        if text in images:
            state['current_id'] = text
            render(load_new=True)
        else:
            print(f"[WARNING] image_id '{text}' not found — staying on {state['current_id']}")

    def reroll(_=None):
        render(load_new=False)

    def save(_=None):
        # Upscale all small panels to match raw size for a uniform grid
        raw_np = state['raw'].numpy()
        H, W = raw_np.shape[:2]

        def to_raw_size(t):
            arr = tf.image.resize(t, [H, W], method='nearest').numpy()
            return np.clip(arr, 0, 1)

        prep_big = to_raw_size(state['preprocessed'])
        aug_a = to_raw_size(state['aug_a'])
        aug_b = to_raw_size(state['aug_b'])

        save_fig, save_axes = plt.subplots(2, 2, figsize=(8, 6))
        save_fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0,
                                  hspace=0.18, wspace=0.02)

        panels = [
            (raw_np, f"Raw {tuple(state['raw'].shape)}"),
            (prep_big, f"Cropped + resized {tuple(state['preprocessed'].shape)}"),
            (aug_a, "Augmentation A"),
            (aug_b, "Augmentation B"),
        ]
        for ax, (img, title) in zip(save_axes.flat, panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(out_dir, f"aug_grid_{state['current_id']}.png")
        save_fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
        plt.close(save_fig)
        print(f"[INFO] Saved {out_path}")

    # Widgets — keep references to prevent garbage collection
    ax_text = plt.axes([0.08, 0.04, 0.12, 0.06])
    text_box = TextBox(ax_text, 'image_id ', initial=state['current_id'])
    text_box.on_submit(jump_to)

    ax_cycle = plt.axes([0.30, 0.04, 0.15, 0.06])
    btn_cycle = Button(ax_cycle, 'Cycle Random')
    btn_cycle.on_clicked(cycle)

    ax_reroll = plt.axes([0.48, 0.04, 0.15, 0.06])
    btn_reroll = Button(ax_reroll, 'Re-roll Augs')
    btn_reroll.on_clicked(reroll)

    ax_save = plt.axes([0.66, 0.04, 0.15, 0.06])
    btn_save = Button(ax_save, 'Save Figure', color='lightgreen')
    btn_save.on_clicked(save)

    fig._widgets = [text_box, btn_cycle, btn_reroll, btn_save]

    render(load_new=True)
    plt.show()


if __name__ == "__main__":
    main()