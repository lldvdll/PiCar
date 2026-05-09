"""
interpret.py — interactive interpretability tool.

Loads frames from RECORDINGS_DIR, runs the model, and visualizes Grad-CAM,
SmoothGrad, and Occlusion sensitivity for the angle and speed heads.

Slider scrubs frames; checkboxes toggle methods and heads (off = blank panel,
no compute, useful for skipping slow Occlusion pass).
"""
import os, glob
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

# --- CONFIG ---
PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RECORDINGS_DIR = os.path.join(PROJECT_PATH, "data", "test_data", "test_data")
RECORDINGS_DIR = os.path.join(PROJECT_PATH, "data", "bad_behaviour", "recordings", "57_tpu_rec", "tbd")
MODEL_NAME = "58_57_more_lighting_aug"
MODEL_PATH = os.path.join(PROJECT_PATH, "src", "picar_autopilot_models",
                          MODEL_NAME, "best_model.h5")
CROP_TOP, CROP_BOTTOM = 120, 30
INPUT_H, INPUT_W = 96, 160

SMOOTHGRAD_N = 15
SMOOTHGRAD_NOISE = 0.10
OCC_PATCH = 16
OCC_STRIDE = 8

# --- LOAD MODEL ---
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

def find_target_layer(m):
    """Last 4D-output layer before GAP — the Grad-CAM target."""
    for layer in reversed(m.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except AttributeError:
            continue
    raise ValueError("No 4D layer found")

TARGET = find_target_layer(model)
print(f"[INFO] Grad-CAM target: {TARGET}")

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(TARGET).output] + list(model.outputs)
)

# --- PREPROCESS (matches model.py) ---
def preprocess(image_bgr):
    im = tf.cast(image_bgr, tf.float32) / 255.0
    im = im[CROP_TOP:-CROP_BOTTOM, :, :]
    im = tf.image.resize(im, [INPUT_H, INPUT_W])
    return tf.expand_dims(im, axis=0)

# --- METHODS (each returns 2D heatmap in [0,1] at INPUT_H × INPUT_W) ---
def gradcam(inp, head):
    with tf.GradientTape() as tape:
        outs = grad_model(inp, training=False)
        conv_out, target = outs[0], outs[1 + head][0, 0]
    grads = tape.gradient(target, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()
    cam = cv2.resize(cam, (INPUT_W, INPUT_H))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def smoothgrad(inp, head):
    accum = tf.zeros_like(inp)
    for _ in range(SMOOTHGRAD_N):
        noisy = inp + tf.random.normal(inp.shape, stddev=SMOOTHGRAD_NOISE)
        with tf.GradientTape() as tape:
            tape.watch(noisy)
            target = model(noisy, training=False)[head][0, 0]
        accum += tf.abs(tape.gradient(target, noisy))
    sal = accum[0].numpy().mean(axis=-1)
    return (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

def occlusion(inp, head):
    baseline = float(model(inp, training=False)[head][0, 0])
    img = inp.numpy()[0]
    heat = np.zeros((INPUT_H, INPUT_W), dtype=np.float32)
    counts = np.zeros_like(heat)
    for y in range(0, INPUT_H - OCC_PATCH + 1, OCC_STRIDE):
        for x in range(0, INPUT_W - OCC_PATCH + 1, OCC_STRIDE):
            occ = img.copy()
            occ[y:y+OCC_PATCH, x:x+OCC_PATCH, :] = 0.5
            pred = float(model(tf.expand_dims(occ, 0), training=False)[head][0, 0])
            heat[y:y+OCC_PATCH, x:x+OCC_PATCH] += abs(pred - baseline)
            counts[y:y+OCC_PATCH, x:x+OCC_PATCH] += 1
    heat = heat / np.maximum(counts, 1)
    return (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

# --- LOAD IMAGES ---

def parse_filename(p):
    parts = os.path.splitext(os.path.basename(p))[0].split("_")
    try:    return int(parts[0]), int(parts[1]), int(parts[2])
    except: return 0, 0, 0


def overlay(img, heat):
    cmap = plt.get_cmap("jet")
    return (0.5 * img + 0.5 * (cmap(heat)[:, :, :3] * 255)).astype(np.uint8)


if __name__ == "__main__":
    image_paths = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No PNGs in {RECORDINGS_DIR}")
    print(f"[INFO] Found {len(image_paths)} frames.")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    plt.subplots_adjust(bottom=0.22, left=0.05, right=0.97, top=0.92)

    COL_LABELS = ["Input", "Grad-CAM", "SmoothGrad", "Occlusion"]
    ROW_LABELS = ["Angle head", "Speed head"]

    ax_slider = plt.axes([0.10, 0.10, 0.55, 0.03])
    slider = Slider(ax_slider, "Frame", 0, len(image_paths) - 1, valinit=0, valstep=1)

    ax_methods = plt.axes([0.70, 0.02, 0.13, 0.13])
    methods_check = CheckButtons(ax_methods, ["Grad-CAM", "SmoothGrad", "Occlusion"],
                                 [True, True, False])
    ax_heads = plt.axes([0.85, 0.02, 0.13, 0.13])
    heads_check = CheckButtons(ax_heads, ["Angle", "Speed"], [True, True])

    state = {"gradcam": True, "smoothgrad": True, "occlusion": False,
             "angle": True, "speed": True}

    def render(_=None):
        idx = int(slider.val)
        raw = cv2.imread(image_paths[idx])
        if raw is None: return
        inp = preprocess(raw)
        inp_disp = (inp[0].numpy() * 255).astype(np.uint8)

        preds = model(inp, training=False)
        pa = float(preds[0][0, 0]) * 80.0 + 50.0
        ps = int(round(float(preds[1][0, 0]))) * 35
        _, ra, rs = parse_filename(image_paths[idx])
        fig.suptitle(f"Frame {idx+1}/{len(image_paths)}  |  recorded {ra}/{rs}  |  predicted {pa:.1f}/{ps}")

        blank = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
        head_keys = ["angle", "speed"]
        method_keys = ["gradcam", "smoothgrad", "occlusion"]
        method_fns = [gradcam, smoothgrad, occlusion]

        for r in range(2):
            for c in range(4):
                ax = axes[r][c]
                ax.clear(); ax.set_xticks([]); ax.set_yticks([])
                if r == 0: ax.set_title(COL_LABELS[c], fontsize=10)
                if c == 0: ax.set_ylabel(ROW_LABELS[r])

                head_on = state[head_keys[r]]
                if c == 0:
                    ax.imshow(inp_disp if head_on else blank)
                else:
                    if head_on and state[method_keys[c - 1]]:
                        h = method_fns[c - 1](inp, r)
                        ax.imshow(overlay(inp_disp, h))
                    else:
                        ax.imshow(blank)
        fig.canvas.draw_idle()

    def on_method(label):
        state[{"Grad-CAM": "gradcam", "SmoothGrad": "smoothgrad", "Occlusion": "occlusion"}[label]] ^= True
        render()

    def on_head(label):
        state[{"Angle": "angle", "Speed": "speed"}[label]] ^= True
        render()

    slider.on_changed(render)
    methods_check.on_clicked(on_method)
    heads_check.on_clicked(on_head)
    render()
    plt.show()