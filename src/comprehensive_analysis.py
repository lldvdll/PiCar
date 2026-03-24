"""
analyse_model.py  —  Unified PiCar model analysis tool.

Usage:
    python analyse_model.py
    (edit EXPERIMENT / SORT_MODE constants at the top of the file)

Layout:
    TOP ROW   : Original | Network Input | Angle overlay | Speed overlay | Info panel
    MID PANEL : 4x8 grid - toggles between KERNELS and ACTIVATIONS mode (hidden by default)
    BUTTONS   : Prev | Next | Flag | Panel | Mode | Prev Layer | Next Layer | Page

Separate diagnostics window shows:
    - Mean angle error heatmap over (true_angle x true_speed) for train and val sets
    - Single-image forward-pass latency histogram
"""

import os
import json
import time
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from sklearn.model_selection import train_test_split


# ===============================================================
#  CONSTANTS  - edit these instead of using command-line args
# ===============================================================

EXPERIMENT  = "22_remove_sqrt_in_balance"   # Experiment folder name inside experiments/

# Sorting / filtering mode for the worst-predictions list.
# Options:
#   "both"   - total angle + speed error  (default)
#   "angle"  - worst angle error only
#   "speed"  - worst speed error only
#   "left"   - worst angle error where true_angle < 0.5  (turning left)
#   "right"  - worst angle error where true_angle >= 0.5 (turning right)
SORT_MODE   = "both"


# ===============================================================
#  EXPERIMENT LOADING
# ===============================================================

def load_experiment(exp_name):
    exp_dir     = os.path.join("experiments", exp_name)
    config_path = os.path.join(exp_dir, "experiment_details.json")
    model_path  = os.path.join(exp_dir, "best_model.h5")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing config or model in: {exp_dir}")

    config = json.load(open(config_path))
    print(f"[INFO] Loading model from {exp_dir} ...")
    model = tf.keras.models.load_model(model_path)

    best_epoch, best_val = "N/A", "N/A"
    log_path = os.path.join("experiments", "model_log.csv")
    if os.path.exists(log_path):
        log  = pd.read_csv(log_path)
        rows = log[log["Experiment_Name"] == exp_name]
        if not rows.empty:
            best_epoch = rows.iloc[-1]["Best_Epoch"]
            best_val   = round(rows.iloc[-1]["Best_Val_Loss"], 5)

    return config, model, best_epoch, best_val


# ===============================================================
#  DATA SPLITS
#  Exactly mirrors prepare_data_pipelines() in train_baseline_wandb.py:
#    1. Load train_clean_weighted.csv
#    2. Drop bad images
#    3. train_test_split(test_size=0.2, random_state=42)  <- split FIRST
#    4. Resample train only (val is never resampled)
# ===============================================================

def get_train_val_dfs(config):
    """
    Reconstructs the exact same train/val DataFrames used during training.
    Bad images are excluded. Val set is the untouched 20% split.
    Returns (train_df, val_df) with original (unoversampled) rows.
    """
    df = pd.read_csv(config["TRAIN_CSV"])
    bad = set(pd.read_csv(config["BAD_IMG_CSV"])["filename"].astype(str).tolist())

    df["_chk"] = df["image_id"].astype(float).astype(int).astype(str) + ".png"
    df = df[~df["_chk"].isin(bad)].drop(columns=["_chk"])
    
    # Apply label corrections before splitting so both sets see corrected labels
    corrections_path = os.path.join("data", "corrections.csv")
    if os.path.exists(corrections_path):
        corr = pd.read_csv(corrections_path, usecols=["file", "variable", "value"])
        corr["_chk"] = corr["file"].astype(str) + ".png"
        for _, row in corr.iterrows():
            mask = df["image_id"].astype(float).astype(int).astype(str) + ".png" == row["_chk"]
            df.loc[mask, row["variable"]] = row["value"]
        print(f"[INFO] Applied {len(corr)} label corrections from {corrections_path}")

    # Exact same split as training
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    val_df   = val_df.drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    return train_df, val_df


# ===============================================================
#  IMAGE LOADING
#  Exactly mirrors read_and_decode_image() + reshape_image() in
#  train_baseline_wandb.py. No augmentation applied (inference only).
# ===============================================================

def load_image(path, config):
    """
    Returns:
        img_net : (H, W, 3) float32 tensor - cropped & resized network input
        img_raw : (H_orig, W_orig, 3) uint8 ndarray - original for display
    """
    raw     = tf.io.read_file(path)
    img_raw = tf.image.decode_png(raw, channels=3).numpy()

    img = tf.cast(tf.image.decode_png(raw, channels=3), tf.float32) / 255.0

    # Exactly matches reshape_image() in training code
    top = config.get("CROP_TOP_PIXELS", 0)
    bot = config.get("CROP_BOTTOM_PIXELS", 0)
    img = img[top : (None if bot == 0 else -bot), :, :]
    img = tf.image.resize(img, [config["IMG_HEIGHT_TARGET"], config["IMG_WIDTH_TARGET"]])
    return img, img_raw


# ===============================================================
#  MODEL INTROSPECTION HELPERS
# ===============================================================

def get_base_model(model):
    for l in model.layers:
        if isinstance(l, tf.keras.Model):
            return l
    raise RuntimeError("No sub-model found inside the outer model.")


def cam_layer_options(base_model):
    """Shallow->deep candidate layers. Shallower = higher spatial res."""
    candidates = [
        "block_3_expand_relu",
        "block_6_expand_relu",
        "block_10_expand_relu",
        "block_13_expand_relu",
        "out_relu",
    ]
    existing = {l.name for l in base_model.layers}
    return [c for c in candidates if c in existing]


def conv_layer_options(base_model):
    """Spatial Conv2D layers only (kernel > 1x1) - suitable for kernel vis."""
    return [
        l.name for l in base_model.layers
        if isinstance(l, tf.keras.layers.Conv2D) and l.kernel.shape[0] > 1
    ]


# ===============================================================
#  GRAD-CAM
# ===============================================================

def build_grad_cam_model(model, base_model):
    """
    Taps the outer functional graph right after the base model output.
    Avoids the 'Graph disconnected' error from nested branching models.
    """
    layer_names     = [l.name for l in model.layers]
    base_idx        = layer_names.index(base_model.name)
    base_out_tensor = model.layers[base_idx + 1].input

    return tf.keras.Model(
        inputs  = model.inputs,
        outputs = [base_out_tensor, model.output[0], model.output[1]],
        name    = "grad_cam_model",
    )


def compute_cams(img_t, gcam_model):
    H, W = img_t.shape[1], img_t.shape[2]

    with tf.GradientTape(persistent=True) as tape:
        features, a_out, s_out = gcam_model(img_t, training=False)
        a_val = a_out[0, 0]
        s_val = s_out[0, 0]

    def _heatmap(scalar_pred):
        grads   = tape.gradient(scalar_pred, features)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam     = tf.reduce_sum(features[0] * weights, axis=-1)
        cam     = tf.maximum(cam, 0.0)
        cam     = cam / (tf.reduce_max(cam) + 1e-10)
        return tf.image.resize(cam[..., tf.newaxis], [H, W])[:, :, 0].numpy()

    cam_a = _heatmap(a_val)
    cam_s = _heatmap(s_val)
    del tape
    return cam_a, cam_s, a_val.numpy(), s_val.numpy()


# ===============================================================
#  SMOOTHGRAD SALIENCY
# ===============================================================

def smooth_grad(img_t, model, output_idx, n=10, noise=0.07):
    """
    Average |gradient| over n noisy copies. Suppresses texture-edge noise
    that makes vanilla gradients highlight irrelevant regions.
    """
    accum = tf.zeros_like(img_t)
    for _ in range(n):
        noisy = tf.clip_by_value(
            img_t + tf.random.normal(img_t.shape, stddev=noise), 0.0, 1.0
        )
        with tf.GradientTape() as tape:
            tape.watch(noisy)
            pred = model(noisy, training=False)[output_idx][0, 0]
        accum += tf.abs(tape.gradient(pred, noisy))

    sal = tf.reduce_max(accum / n, axis=-1)[0]
    return ((sal - tf.reduce_min(sal)) / (tf.reduce_max(sal) - tf.reduce_min(sal) + 1e-10)).numpy()


# ===============================================================
#  FEATURE MAP ACTIVATIONS
# ===============================================================

def get_feature_maps(img_t, base_model, layer_name):
    extractor = tf.keras.Model(
        inputs  = base_model.inputs,
        outputs = base_model.get_layer(layer_name).output,
    )
    return extractor.predict(img_t, verbose=0)[0]


# ===============================================================
#  KERNEL VISUALISATION
# ===============================================================

def get_kernels(base_model, layer_name):
    return base_model.get_layer(layer_name).get_weights()[0]


def render_kernel(kernel):
    k = kernel.copy().astype(np.float32)
    if k.shape[2] == 3:
        k_norm = (k - k.min()) / (k.max() - k.min() + 1e-10)
    else:
        k      = k.mean(axis=2, keepdims=True)
        k_norm = (k - k.min()) / (k.max() - k.min() + 1e-10)
        k_norm = np.repeat(k_norm, 3, axis=2)
    return tf.image.resize(k_norm, [28, 28], method="nearest").numpy()


# ===============================================================
#  BATCH INFERENCE
#  Runs on every image in both train_df and val_df.
#  Timings collected from the first 200 val images at batch=1.
# ===============================================================

def run_inference(train_df, val_df, model, config):
    def _score_df(df, label):
        print(f"[INFO] Running inference on {label} set ({len(df)} images) ...")
        paths   = df["filepath"].values
        tensors = [load_image(p, config)[0] for p in paths]
        batch   = tf.convert_to_tensor(tensors)
        preds   = model.predict(batch, verbose=1)

        results = []
        for i in range(len(df)):
            ta = float(df.iloc[i]["angle"])
            ts = float(df.iloc[i]["speed"])
            pa = float(preds[0][i][0])
            ps = float(preds[1][i][0])
            ea, es = abs(ta - pa), abs(ts - ps)
            results.append(dict(
                filepath  = paths[i],
                true_a=ta, pred_a=pa, err_a=ea,
                true_s=ts, pred_s=ps, err_s=es,
                total_err = ea + es,
            ))
        return results

    train_results = _score_df(train_df, "train")
    val_results   = _score_df(val_df,   "val")

    # Time individual forward passes on val set (batch=1). First call is
    # GPU warm-up; discarded so JIT compilation doesn't skew the distribution.
    print("[INFO] Timing individual forward passes on val set ...")
    val_tensors = [load_image(p, config)[0] for p in val_df["filepath"].values]
    _ = model(tf.expand_dims(val_tensors[0], 0), training=False)  # warm-up
    runtimes_ms = []
    for i in range(min(200, len(val_tensors))):
        t0 = time.perf_counter()
        model(tf.expand_dims(val_tensors[i], 0), training=False)
        runtimes_ms.append((time.perf_counter() - t0) * 1000)

    # Sort val results for the interactive viewer
    mode = SORT_MODE.lower()
    if mode == "angle":
        sorted_val = sorted(val_results, key=lambda x: x["err_a"], reverse=True)
    elif mode == "speed":
        sorted_val = sorted(val_results, key=lambda x: x["err_s"], reverse=True)
    elif mode == "left":
        sorted_val = sorted([r for r in val_results if r["true_a"] < 0.5],
                            key=lambda x: x["err_a"], reverse=True)
    elif mode == "right":
        sorted_val = sorted([r for r in val_results if r["true_a"] >= 0.5],
                            key=lambda x: x["err_a"], reverse=True)
    else:
        sorted_val = sorted(val_results, key=lambda x: x["total_err"], reverse=True)

    return train_results, val_results, sorted_val, runtimes_ms


# ===============================================================
#  FLAG HELPER
# ===============================================================

def flag_image(filepath, bad_csv_path):
    fname = os.path.basename(filepath)
    existing = set()
    if os.path.exists(bad_csv_path):
        try:
            existing = set(pd.read_csv(bad_csv_path)["filename"].astype(str).tolist())
        except Exception:
            pass

    if fname in existing:
        print(f"[INFO] Already flagged: {fname}")
        return False

    write_mode = "a" if os.path.exists(bad_csv_path) else "w"
    with open(bad_csv_path, write_mode) as f:
        if write_mode == "w":
            f.write("filename\n")
        f.write(fname + "\n")
    print(f"[INFO] Flagged: {fname}")
    return True


# ===============================================================
#  DIAGNOSTICS FIGURE  (separate window)
# ===============================================================

def show_diagnostics(train_results, val_results, runtimes_ms):
    """
    Three-panel diagnostics window:
      Left   - mean angle error heatmap for the TRAINING set
      Centre - mean angle error heatmap for the VALIDATION set
      Right  - forward-pass latency histogram (val set, batch=1)

    Keeps the window open without blocking the main viewer.
    plt.pause() is used instead of plt.show() so it doesn't block.
    """
    def _pivot(results):
        df = pd.DataFrame(results)
        df["angle_bin"] = df["true_a"].round(2)
        df["speed_bin"] = df["true_s"].round(2)
        return df.groupby(["angle_bin", "speed_bin"])["err_a"].mean().unstack(fill_value=np.nan)

    train_pivot = _pivot(train_results)
    val_pivot   = _pivot(val_results)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.patch.set_facecolor("#0e0e16")
    fig.canvas.manager.set_window_title("PiCar - Diagnostics")
    for ax in axes:
        ax.set_facecolor("#0e0e16")
        ax.tick_params(colors="#9090bb")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a44")

    heatmap_kw = dict(cmap="YlOrRd", linewidths=0.3, linecolor="#0e0e16",
                      cbar_kws={"label": "Mean |angle error|"})

    sns.heatmap(train_pivot, ax=axes[0], **heatmap_kw)
    axes[0].set_title("Train — Mean Angle Error by (Angle, Speed)",
                       color="#c8c8ee", fontsize=10, pad=8)
    axes[0].set_xlabel("True Speed (binned)", color="#9090bb")
    axes[0].set_ylabel("True Angle (binned)", color="#9090bb")
    axes[0].tick_params(axis="x", rotation=45, labelsize=7)
    axes[0].tick_params(axis="y", rotation=0,  labelsize=7)

    sns.heatmap(val_pivot, ax=axes[1], **heatmap_kw)
    axes[1].set_title("Val — Mean Angle Error by (Angle, Speed)",
                       color="#c8c8ee", fontsize=10, pad=8)
    axes[1].set_xlabel("True Speed (binned)", color="#9090bb")
    axes[1].set_ylabel("True Angle (binned)", color="#9090bb")
    axes[1].tick_params(axis="x", rotation=45, labelsize=7)
    axes[1].tick_params(axis="y", rotation=0,  labelsize=7)

    rms = np.array(runtimes_ms)
    axes[2].hist(rms, bins=40, color="#2a5080", edgecolor="#0e0e16")
    axes[2].axvline(np.median(rms), color="#e07040", linewidth=1.5,
                    label=f"Median  {np.median(rms):.1f} ms")
    axes[2].axvline(np.percentile(rms, 95), color="#e0d040", linewidth=1.5,
                    linestyle="--", label=f"P95  {np.percentile(rms, 95):.1f} ms")
    axes[2].set_title("Forward Pass Latency  (val set, batch=1, n=200)",
                       color="#c8c8ee", fontsize=10, pad=8)
    axes[2].set_xlabel("Latency (ms)", color="#9090bb")
    axes[2].set_ylabel("Count",        color="#9090bb")
    axes[2].legend(facecolor="#1a1a2e", labelcolor="#c8c8ee", fontsize=9)

    plt.tight_layout()
    # pause() renders the figure and returns immediately — the window stays
    # open while the main interactive viewer runs its own plt.show() loop.
    plt.pause(0.1)


# ===============================================================
#  MAIN — INTERACTIVE VIEWER
# ===============================================================

def main():
    config, model, best_epoch, best_val = load_experiment(EXPERIMENT)

    print("[INFO] Reconstructing train/val splits (exact match to training) ...")
    train_df, val_df = get_train_val_dfs(config)
    print(f"[INFO] Train: {len(train_df)} images | Val: {len(val_df)} images")

    train_results, val_results, sorted_val, runtimes_ms = run_inference(
        train_df, val_df, model, config
    )
    print(f"[INFO] Inference complete. Sorted by: {SORT_MODE}. Showing {len(sorted_val)} images.")

    # Show diagnostics in a separate non-blocking window before the main viewer
    show_diagnostics(train_results, val_results, runtimes_ms)

    base_model  = get_base_model(model)
    cam_opts    = cam_layer_options(base_model)
    conv_opts   = conv_layer_options(base_model)
    gcam_model  = build_grad_cam_model(model, base_model)

    if not cam_opts:
        raise RuntimeError("No CAM candidate layers found — is this a MobileNetV2 model?")

    results = sorted_val   # shorthand for the interactive viewer

    # ── Shared state ──────────────────────────────────────────
    S = {
        "idx"          : 0,
        "panel_visible": False,
        "mode"         : "kernels",
        "conv_idx"     : 0,
        "act_idx"      : 1,
        "feat_page"    : 0,
        "flagged"      : set(),
        "feat_cache"   : None,
        "last"         : None,
    }

    PANEL_R, PANEL_C = 4, 8
    N_PANEL = PANEL_R * PANEL_C

    BG      = "#0e0e16"
    CELL_BG = "#08080f"
    TXT     = "#c8c8ee"
    BORDER  = "#2a2a44"

    # ── Figure layout ─────────────────────────────────────────
    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor(BG)
    fig.canvas.manager.set_window_title(
        f"PiCar Model Analysis — {EXPERIMENT}  [{SORT_MODE}]"
    )

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[3.0, 4.8, 0.6],
        hspace=0.10,
        left=0.01, right=0.99, top=0.97, bottom=0.01,
    )

    top_gs  = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[0], wspace=0.04)
    ax_orig  = fig.add_subplot(top_gs[0, 0:2])
    ax_crop  = fig.add_subplot(top_gs[0, 2])
    ax_angle = fig.add_subplot(top_gs[0, 3])
    ax_speed = fig.add_subplot(top_gs[0, 4])
    ax_info  = fig.add_subplot(top_gs[0, 5])

    mid_gs = gridspec.GridSpecFromSubplotSpec(
        PANEL_R, PANEL_C, subplot_spec=outer[1], wspace=0.03, hspace=0.03
    )
    panel_axes = [
        fig.add_subplot(mid_gs[r, c])
        for r in range(PANEL_R) for c in range(PANEL_C)
    ]

    # Panel title text object — created BEFORE the hide loop
    panel_title = fig.text(
        0.50, 0.412, "",
        ha="center", va="bottom",
        color="#9090bb", fontsize=8.5, family="monospace",
    )

    # Hide panel by default
    for ax in panel_axes:
        ax.set_visible(False)
    panel_title.set_visible(False)

    # Style all axes
    for ax in [ax_orig, ax_crop, ax_angle, ax_speed, ax_info] + panel_axes:
        ax.set_facecolor(CELL_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    # ── Buttons ───────────────────────────────────────────────
    btn_gs  = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=outer[2], wspace=0.07)
    BTN_DEF = [
        ("Prev",         "#1b3356", "#2a5080"),
        ("Next",         "#1b3356", "#2a5080"),
        ("Flag",         "#4a1010", "#7a2020"),
        ("Panel: Off",   "#2a1a3a", "#4a2a5a"),
        ("Mode",         "#1a3020", "#2a5030"),
        ("Prev Layer",   "#22223a", "#3a3a60"),
        ("Next Layer",   "#22223a", "#3a3a60"),
        ("Next Page",    "#22223a", "#3a3a60"),
    ]
    btn_axes = [fig.add_subplot(btn_gs[0, i]) for i in range(8)]
    btns = [
        Button(ax, lbl, color=col, hovercolor=hov)
        for ax, (lbl, col, hov) in zip(btn_axes, BTN_DEF)
    ]
    for b in btns:
        b.label.set_color("white")
        b.label.set_fontsize(9)

    (btn_prev, btn_next, btn_flag,
     btn_panel, btn_mode, btn_pl, btn_nl, btn_paging) = btns

    # ── Render helpers ────────────────────────────────────────

    def _render_top(item, img_net, img_raw, cam_a, cam_s, sal_a, sal_s):
        ax_orig.clear()
        ax_orig.imshow(img_raw)
        ax_orig.axis("off")
        ax_orig.set_title("Original", color=TXT, fontsize=8, pad=2)

        ax_crop.clear()
        ax_crop.imshow(img_net.numpy())
        ax_crop.axis("off")
        ax_crop.set_title("Network Input", color=TXT, fontsize=8, pad=2)

        for ax, cam, sal, label in (
            (ax_angle, cam_a, sal_a, "Angle"),
            (ax_speed, cam_s, sal_s, "Speed"),
        ):
            ax.clear()
            ax.imshow(img_net.numpy())
            ax.imshow(sal, cmap="magma", alpha=0.38, vmin=0, vmax=1)
            ax.imshow(cam, cmap="jet",   alpha=0.40, vmin=0, vmax=1)
            ax.axis("off")
            ax.set_title(f"{label} — SmoothGrad + Grad-CAM", color=TXT, fontsize=7.5, pad=2)

        flag_str = "  FLAGGED" if item["filepath"] in S["flagged"] else ""
        ax_info.clear()
        ax_info.axis("off")
        info = (
            f"Rank #{S['idx'] + 1} worst{flag_str}\n"
            f"{os.path.basename(item['filepath'])}\n"
            f"{'─' * 26}\n"
            f" ANGLE  true {item['true_a']:.3f}  pred {item['pred_a']:.3f}\n"
            f"        err  {item['err_a']:.3f}\n\n"
            f" SPEED  true {item['true_s']:.3f}  pred {item['pred_s']:.3f}\n"
            f"        err  {item['err_s']:.3f}\n\n"
            f" total err   {item['total_err']:.4f}\n"
            f"{'─' * 26}\n"
            f" Exp   {EXPERIMENT}  [{SORT_MODE}]\n"
            f" Epoch {best_epoch}  Val {best_val}"
        )
        ax_info.text(
            0.05, 0.97, info,
            transform=ax_info.transAxes,
            fontsize=8, va="top", family="monospace", color="#d0d0ff",
            bbox=dict(facecolor="#0e0e2a", alpha=0.92, edgecolor="#3c3c88",
                      boxstyle="round,pad=0.55"),
        )

    def _render_panel():
        item = results[S["idx"]]

        if S["mode"] == "kernels":
            layer   = conv_opts[S["conv_idx"]]
            kernels = get_kernels(base_model, layer)
            kH, kW, in_ch, n_filt = kernels.shape
            panel_title.set_text(
                f"KERNELS  —  {layer}   "
                f"({kH}x{kW} kernel  in_ch={in_ch}  n_filters={n_filt})"
            )
            for i, ax in enumerate(panel_axes):
                ax.clear(); ax.axis("off")
                if i < n_filt:
                    ax.imshow(render_kernel(kernels[:, :, :, i]))
                    ax.set_title(str(i), fontsize=5, color="#6a6a88", pad=1)
        else:
            layer = cam_opts[S["act_idx"]]
            if S["feat_cache"] is None or S["feat_cache"][0] != layer:
                img_t = tf.expand_dims(load_image(item["filepath"], config)[0], 0)
                S["feat_cache"] = (layer, get_feature_maps(img_t, base_model, layer))

            fm      = S["feat_cache"][1]
            n_total = fm.shape[-1]
            start   = S["feat_page"] * N_PANEL
            end     = min(start + N_PANEL, n_total)
            panel_title.set_text(
                f"ACTIVATIONS  —  {layer}   "
                f"(features {start}-{end - 1} of {n_total}  |  page {S['feat_page'] + 1})"
            )
            for i, ax in enumerate(panel_axes):
                ax.clear(); ax.axis("off")
                fi = start + i
                if fi < n_total:
                    ax.imshow(fm[:, :, fi], cmap="viridis", aspect="auto")
                    ax.set_title(str(fi), fontsize=5, color="#6a6a88", pad=1)

    def full_update():
        item = results[S["idx"]]
        print(f"[INFO] Analysing rank #{S['idx'] + 1}: {os.path.basename(item['filepath'])}")

        img_net, img_raw = load_image(item["filepath"], config)
        img_t = tf.expand_dims(img_net, 0)

        cam_a, cam_s, _, _ = compute_cams(img_t, gcam_model)
        sal_a = smooth_grad(img_t, model, 0)
        sal_s = smooth_grad(img_t, model, 1)

        S["last"]       = (img_net, img_raw, cam_a, cam_s, sal_a, sal_s)
        S["feat_cache"] = None

        _render_top(item, *S["last"])
        if S["panel_visible"]:
            _render_panel()
        fig.canvas.draw_idle()

    def panel_only():
        _render_panel()
        fig.canvas.draw_idle()

    def top_only():
        if S["last"] is not None:
            _render_top(results[S["idx"]], *S["last"])
            fig.canvas.draw_idle()

    # ── Button callbacks ──────────────────────────────────────

    def on_prev(_):
        S["idx"]       = (S["idx"] - 1) % len(results)
        S["feat_page"] = 0
        full_update()

    def on_next(_):
        S["idx"]       = (S["idx"] + 1) % len(results)
        S["feat_page"] = 0
        full_update()

    def on_flag(_):
        item    = results[S["idx"]]
        flagged = flag_image(item["filepath"], config["BAD_IMG_CSV"])
        if flagged:
            S["flagged"].add(item["filepath"])
        top_only()

    def on_panel_toggle(_):
        S["panel_visible"] = not S["panel_visible"]
        label = "Panel: On" if S["panel_visible"] else "Panel: Off"
        btn_panel.label.set_text(label)
        for ax in panel_axes:
            ax.set_visible(S["panel_visible"])
        panel_title.set_visible(S["panel_visible"])
        if S["panel_visible"]:
            panel_only()
        else:
            fig.canvas.draw_idle()

    def on_mode(_):
        S["mode"]      = "activations" if S["mode"] == "kernels" else "kernels"
        S["feat_page"] = 0
        btn_mode.label.set_text(
            "Mode: Kernels" if S["mode"] == "kernels" else "Mode: Activ."
        )
        if S["panel_visible"]:
            panel_only()

    def on_prev_layer(_):
        if S["mode"] == "kernels":
            S["conv_idx"] = (S["conv_idx"] - 1) % len(conv_opts)
        else:
            S["act_idx"]    = (S["act_idx"] - 1) % len(cam_opts)
            S["feat_cache"] = None
        S["feat_page"] = 0
        if S["panel_visible"]:
            panel_only()

    def on_next_layer(_):
        if S["mode"] == "kernels":
            S["conv_idx"] = (S["conv_idx"] + 1) % len(conv_opts)
        else:
            S["act_idx"]    = (S["act_idx"] + 1) % len(cam_opts)
            S["feat_cache"] = None
        S["feat_page"] = 0
        if S["panel_visible"]:
            panel_only()

    def on_paging(_):
        if S["mode"] == "activations" and S["feat_cache"] is not None:
            n_total  = S["feat_cache"][1].shape[-1]
            max_page = (n_total - 1) // N_PANEL
            S["feat_page"] = (S["feat_page"] + 1) % (max_page + 1)
            if S["panel_visible"]:
                panel_only()

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_flag.on_clicked(on_flag)
    btn_panel.on_clicked(on_panel_toggle)
    btn_mode.on_clicked(on_mode)
    btn_pl.on_clicked(on_prev_layer)
    btn_nl.on_clicked(on_next_layer)
    btn_paging.on_clicked(on_paging)

    full_update()
    # plt.show() blocks here and keeps BOTH windows alive
    plt.show()


if __name__ == "__main__":
    main()