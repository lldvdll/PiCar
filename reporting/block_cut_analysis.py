"""
block_cut_analysis.py — feature map visualisation across three MobileNetV2 blocks
to justify the architecture cut point.

Layout:  raw image | features at block A
                   | features at block B
                   | features at block C
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, TextBox
import tensorflow as tf

# --- PATHS & PREPROCESSING (inlined from training script) ---
PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAIN_CSV = os.path.join(PROJECT_PATH, "data", "train.csv")
TRAIN_IMG_DIR = os.path.join(PROJECT_PATH, "data", "training_data", "training_data")
MODEL_PATH = os.path.join(PROJECT_PATH, "experiments", "14_unfreeze_by_block", "best_model.h5")

CROP_TOP, CROP_BOTTOM = 120, 30
INPUT_H, INPUT_W = 96, 160
N_FEATURES_PER_BLOCK = 4
DEFAULT_BLOCKS = ["block_6_expand_relu", "block_10_expand_relu", "block_14_expand_relu"]


def preprocess_image(filepath):
    """Match the training pipeline: decode -> normalize -> crop -> resize."""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = img[CROP_TOP:-CROP_BOTTOM, :, :]
    img = tf.image.resize(img, [INPUT_H, INPUT_W])
    return img


def load_index(csv_path, img_dir):
    print(f"[INFO] Loading index from {csv_path}...")
    df = pd.read_csv(csv_path)
    items = {}
    for _, row in df.dropna(subset=['image_id']).iterrows():
        img_id = str(int(float(row['image_id'])))
        full_path = os.path.join(img_dir, img_id + '.png').replace("\\", "/")
        if os.path.exists(full_path):
            items[img_id] = {'filepath': full_path}
    print(f"[INFO] Indexed {len(items)} images.")
    return items


def find_base_model(model):
    """The MobileNetV2 backbone is nested as a sub-Model inside the head."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested base model found")


def top_n_features(feature_map, n):
    """Channels with the highest variance — most spatially informative."""
    var = feature_map.var(axis=(0, 1))
    return np.argsort(var)[-n:][::-1]


def main():
    images = load_index(TRAIN_CSV, TRAIN_IMG_DIR)
    if not images:
        raise ValueError("No images found! Check paths.")
    image_ids = list(images.keys())

    print(f"[INFO] Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    base_model = find_base_model(model)
    available_layers = {l.name for l in base_model.layers}
    print(f"[DEBUG] Base model has {len(base_model.layers)} layers; "
          f"last: {[l.name for l in base_model.layers[-3:]]}")

    fig = plt.figure(figsize=(14, 7))
    gs_master = gridspec.GridSpec(1, 5, figure=fig,
                                   width_ratios=[1.2, 1, 1, 1, 1],
                                   wspace=0.05)
    plt.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.16, wspace=0.05)

    # Left: input image, vertically centred
    gs_left = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[0, 0])
    ax_input = fig.add_subplot(gs_left[1, 0])

    # Right: 3 rows × N feature maps. Larger hspace makes room for per-row titles.
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, N_FEATURES_PER_BLOCK,
        subplot_spec=gs_master[0, 1:],
        hspace=0.45, wspace=0.05
    )
    feat_axes = [
        [fig.add_subplot(gs_right[r, c]) for c in range(N_FEATURES_PER_BLOCK)]
        for r in range(3)
    ]

    state = {
        'current_id': random.choice(image_ids),
        'blocks': list(DEFAULT_BLOCKS),
        'img_tensor': None,
    }

    def get_features(layer_name):
        if layer_name not in available_layers:
            return None
        extractor = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=base_model.get_layer(layer_name).output,
        )
        return extractor.predict(state['img_tensor'], verbose=0)[0]

    def render(load_new=True):
        if load_new:
            item = images[state['current_id']]
            img = preprocess_image(item['filepath'])
            state['img_tensor'] = tf.expand_dims(img, 0)

        ax_input.clear(); ax_input.axis('off')
        ax_input.imshow(state['img_tensor'].numpy()[0])
        ax_input.set_title(f"Input (id={state['current_id']})", fontsize=10)

        for r, layer_name in enumerate(state['blocks']):
            feats = get_features(layer_name)
            for c in range(N_FEATURES_PER_BLOCK):
                ax = feat_axes[r][c]
                ax.clear()
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                if feats is None:
                    ax.axis('off')
                    if c == 0:
                        ax.text(0.5, 0.5, f"layer\n'{layer_name}'\nnot found",
                                ha='center', va='center', transform=ax.transAxes,
                                fontsize=9, color='red')
                    continue

                top_idx = top_n_features(feats, N_FEATURES_PER_BLOCK)
                channel = top_idx[c]
                ax.imshow(feats[:, :, channel], cmap='viridis', aspect='auto')

            # Per-row title spans the four panels of that row, sat above
            if feats is not None:
                row_centre_ax = feat_axes[r][N_FEATURES_PER_BLOCK // 2 - 1]
                # Place a title above the first panel in the row, anchored leftward
                feat_axes[r][0].set_title(
                    f"{layer_name}   ({feats.shape[0]}×{feats.shape[1]})",
                    fontsize=10, loc='left'
                )

        fig.canvas.draw_idle()

    def cycle(_=None):
        state['current_id'] = random.choice(image_ids)
        text_id.set_val(state['current_id'])
        render(load_new=True)

    def jump_to(text):
        if text in images:
            state['current_id'] = text
            render(load_new=True)
        else:
            print(f"[WARNING] image_id '{text}' not found")

    def update_block(idx):
        def _set(text):
            text = text.strip()
            if text not in available_layers:
                print(f"[WARNING] layer '{text}' not in base model — kept '{state['blocks'][idx]}'")
                return
            state['blocks'][idx] = text
            render(load_new=False)
        return _set

    def save(_=None):
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(out_dir, f"features_{state['current_id']}.png")
        fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
        print(f"[INFO] Saved {out_path} (includes widget bar — crop in slide tool)")

    ax_id = plt.axes([0.03, 0.04, 0.10, 0.05])
    text_id = TextBox(ax_id, 'image_id ', initial=state['current_id'])
    text_id.on_submit(jump_to)

    ax_b1 = plt.axes([0.20, 0.04, 0.16, 0.05])
    text_b1 = TextBox(ax_b1, 'Block 1 ', initial=state['blocks'][0])
    text_b1.on_submit(update_block(0))

    ax_b2 = plt.axes([0.38, 0.04, 0.16, 0.05])
    text_b2 = TextBox(ax_b2, 'Block 2 ', initial=state['blocks'][1])
    text_b2.on_submit(update_block(1))

    ax_b3 = plt.axes([0.56, 0.04, 0.16, 0.05])
    text_b3 = TextBox(ax_b3, 'Block 3 ', initial=state['blocks'][2])
    text_b3.on_submit(update_block(2))

    ax_cycle = plt.axes([0.75, 0.04, 0.10, 0.05])
    btn_cycle = Button(ax_cycle, 'Cycle Random')
    btn_cycle.on_clicked(cycle)

    ax_save = plt.axes([0.86, 0.04, 0.10, 0.05])
    btn_save = Button(ax_save, 'Save', color='lightgreen')
    btn_save.on_clicked(save)

    fig._widgets = [text_id, text_b1, text_b2, text_b3, btn_cycle, btn_save]

    render(load_new=True)
    plt.show()


if __name__ == "__main__":
    main()