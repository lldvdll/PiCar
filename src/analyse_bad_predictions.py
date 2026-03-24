import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bad Predictions with Saliency & Grad-CAM")
    parser.add_argument("-e", "--experiment", type=str, required=True, help="Name of the experiment folder")
    return parser.parse_args()

def load_experiment_data(exp_name):
    exp_dir = os.path.join("experiments", exp_name)
    config_path = os.path.join(exp_dir, "experiment_details.json")
    model_path = os.path.join(exp_dir, "best_model.h5")
    
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find config or model in {exp_dir}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print(f"[INFO] Loading Model for {exp_name}...")
    model = tf.keras.models.load_model(model_path)
    
    # Get global metrics
    best_epoch, best_val = "N/A", "N/A"
    log_path = os.path.join("experiments", "model_log.csv")
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        exp_logs = log_df[log_df["Experiment_Name"] == exp_name]
        if not exp_logs.empty:
            best_epoch = exp_logs.iloc[-1]["Best_Epoch"]
            best_val = round(exp_logs.iloc[-1]["Best_Val_Loss"], 4)
            
    return config, model, best_epoch, best_val

def get_validation_sample(config, sample_size=500):
    print("[INFO] Reconstructing Validation Set...")
    df = pd.read_csv(config["TRAIN_CSV"])
    bad_df = pd.read_csv(config["BAD_IMG_CSV"])
    
    bad_list = bad_df['filename'].astype(str).tolist()
    df['check_name'] = df['image_id'].astype(float).astype(int).astype(str) + '.png'
    df = df[~df['check_name'].isin(bad_list)].drop(columns=['check_name'])
    
    balanced_df = df.sample(n=len(df), replace=True, weights='sample_weight', random_state=42)
    _, val_df = train_test_split(balanced_df, test_size=0.2, random_state=42)
    val_df = val_df.drop_duplicates(subset=['filepath']).reset_index(drop=True)
    
    # Sample a subset to evaluate quickly
    return val_df.sample(n=min(sample_size, len(val_df)), random_state=42)

def preprocess_for_network(image_path, config):
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_png(img_raw, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    
    top = config.get("CROP_TOP_PIXELS", 0)
    bottom_crop = config.get("CROP_BOTTOM_PIXELS", 0)
    bottom = -bottom_crop if bottom_crop > 0 else None
    
    img_cropped = img[top:bottom, :, :]
    img_resized = tf.image.resize(img_cropped, [config["IMG_HEIGHT_TARGET"], config["IMG_WIDTH_TARGET"]])
    return img_resized, tf.image.decode_png(img_raw, channels=3).numpy()

def generate_maps(img_tensor, model):
    # 1. Dynamically find the base model layer (Backwards compatible)
    layer_names = [l.name for l in model.layers]
    if "Amputated_MobileNetV2" in layer_names:
        base_layer_name = "Amputated_MobileNetV2"
    else:
        # Fallback for older models (like 13_unfreeze_60_500_epochs)
        base_layer_name = next((name for name in layer_names if "mobilenet" in name.lower()), None)
        
    if base_layer_name is None:
        raise ValueError(f"Could not find a base feature extractor in: {layer_names}")

    base_layer_idx = layer_names.index(base_layer_name)
    
    # 2. THE FIX: Grab the tensor from the outer graph by looking at what the NEXT layer consumes.
    # This completely avoids the "Graph disconnected" nested-model error!
    base_output_outer = model.layers[base_layer_idx + 1].input

    # Setup the gradient model to extract the features using the outer tensor
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[base_output_outer, model.output])

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(img_tensor)
        conv_outputs, preds = grad_model(img_tensor)
        angle_pred = preds[0][0, 0]
        speed_pred = preds[1][0, 0]

    # --- Saliency Maps ---
    sal_angle = tape.gradient(angle_pred, img_tensor)
    sal_speed = tape.gradient(speed_pred, img_tensor)
    
    # Reduce across channels and normalize
    sal_angle = tf.reduce_max(tf.abs(sal_angle), axis=-1)[0]
    sal_angle = (sal_angle - tf.reduce_min(sal_angle)) / (tf.reduce_max(sal_angle) - tf.reduce_min(sal_angle) + 1e-10)
    
    sal_speed = tf.reduce_max(tf.abs(sal_speed), axis=-1)[0]
    sal_speed = (sal_speed - tf.reduce_min(sal_speed)) / (tf.reduce_max(sal_speed) - tf.reduce_min(sal_speed) + 1e-10)

    # --- Grad-CAM ---
    grads_angle = tape.gradient(angle_pred, conv_outputs)
    grads_speed = tape.gradient(speed_pred, conv_outputs)
    
    pooled_angle = tf.reduce_mean(grads_angle, axis=(0, 1, 2))
    heatmap_angle = conv_outputs[0] @ pooled_angle[..., tf.newaxis]
    heatmap_angle = tf.maximum(tf.squeeze(heatmap_angle), 0)
    heatmap_angle /= (tf.math.reduce_max(heatmap_angle) + 1e-10)

    pooled_speed = tf.reduce_mean(grads_speed, axis=(0, 1, 2))
    heatmap_speed = conv_outputs[0] @ pooled_speed[..., tf.newaxis]
    heatmap_speed = tf.maximum(tf.squeeze(heatmap_speed), 0)
    heatmap_speed /= (tf.math.reduce_max(heatmap_speed) + 1e-10)
    
    del tape
    
    # Resize heatmaps to match input image
    hm_angle_rs = tf.image.resize(heatmap_angle[..., tf.newaxis], [img_tensor.shape[1], img_tensor.shape[2]])[:, :, 0]
    hm_speed_rs = tf.image.resize(heatmap_speed[..., tf.newaxis], [img_tensor.shape[1], img_tensor.shape[2]])[:, :, 0]

    return sal_angle.numpy(), sal_speed.numpy(), hm_angle_rs.numpy(), hm_speed_rs.numpy(), angle_pred.numpy(), speed_pred.numpy()

def main():
    args = parse_args()
    config, model, best_epoch, best_val = load_experiment_data(args.experiment)
    val_df = get_validation_sample(config)
    
    print("[INFO] Running inference to find worst predictions...")
    results = []
    
    # Process in a single batch for speed
    paths = val_df['filepath'].values
    tensors = []
    for p in paths:
        t, _ = preprocess_for_network(p, config)
        tensors.append(t)
        
    batch_tensors = tf.convert_to_tensor(tensors)
    preds = model.predict(batch_tensors, verbose=1)
    
    for i in range(len(val_df)):
        true_a = val_df.iloc[i]['angle']
        true_s = val_df.iloc[i]['speed']
        pred_a = preds[0][i][0]
        pred_s = preds[1][i][0]
        
        err_a = abs(true_a - pred_a)
        err_s = abs(true_s - pred_s)
        total_err = err_a + err_s
        
        results.append({
            'filepath': paths[i],
            'true_a': true_a, 'pred_a': pred_a, 'err_a': err_a,
            'true_s': true_s, 'pred_s': pred_s, 'err_s': err_s,
            'total_err': total_err
        })
        
    # Sort by total error and take top 50
    results = sorted(results, key=lambda x: x['total_err'], reverse=True)[:50]
    
    print(f"\n[INFO] Displaying top {len(results)} worst predictions.")
    
    # --- UI Setup ---
    fig = plt.figure(figsize=(18, 9))
    fig.canvas.manager.set_window_title(f"Bad Predictions Viewer: {args.experiment}")
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1])
    
    ax_orig = fig.add_subplot(gs[0, 0:2])
    ax_prep = fig.add_subplot(gs[0, 2])
    ax_text = fig.add_subplot(gs[0, 3])
    
    ax_sal_a = fig.add_subplot(gs[1, 0])
    ax_cam_a = fig.add_subplot(gs[1, 1])
    ax_sal_s = fig.add_subplot(gs[1, 2])
    ax_cam_s = fig.add_subplot(gs[1, 3])
    
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.98, top=0.92)
    
    state = {'idx': 0}

    def update_plot():
        item = results[state['idx']]
        img_net, img_raw = preprocess_for_network(item['filepath'], config)
        img_tensor = tf.expand_dims(img_net, 0)
        
        sal_a, sal_s, cam_a, cam_s, p_a, p_s = generate_maps(img_tensor, model)
        
        # Row 1
        ax_orig.clear(); ax_orig.imshow(img_raw); ax_orig.set_title("Original Image"); ax_orig.axis('off')
        ax_prep.clear(); ax_prep.imshow(img_net.numpy()); ax_prep.set_title("Network Input (Cropped)"); ax_prep.axis('off')
        
        ax_text.clear(); ax_text.axis('off')
        info_str = (
            f"EXPERIMENT: {args.experiment}\n"
            f"Image: {item['filepath'].split('/')[-1]}\n"
            f"Best Epoch: {best_epoch} | Global Val Loss: {best_val}\n"
            f"----------------------------------------\n"
            f"Rank: {state['idx'] + 1} Worst Prediction\n\n"
            f"ANGLE:\n"
            f"  True: {item['true_a']:.3f}\n"
            f"  Pred: {item['pred_a']:.3f}\n"
            f"  Error: {item['err_a']:.3f}\n\n"
            f"SPEED:\n"
            f"  True: {item['true_s']:.3f}\n"
            f"  Pred: {item['pred_s']:.3f}\n"
            f"  Error: {item['err_s']:.3f}"
        )
        ax_text.text(0.1, 0.9, info_str, fontsize=12, verticalalignment='top', family='monospace', 
                     bbox=dict(facecolor='lightgray', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))

        # Row 2
        ax_sal_a.clear(); ax_sal_a.imshow(img_net.numpy()); ax_sal_a.imshow(sal_a, cmap='magma', alpha=0.6)
        ax_sal_a.set_title("Angle Saliency Map"); ax_sal_a.axis('off')
        
        ax_cam_a.clear(); ax_cam_a.imshow(img_net.numpy()); ax_cam_a.imshow(cam_a, cmap='jet', alpha=0.5)
        ax_cam_a.set_title("Angle Grad-CAM"); ax_cam_a.axis('off')
        
        ax_sal_s.clear(); ax_sal_s.imshow(img_net.numpy()); ax_sal_s.imshow(sal_s, cmap='magma', alpha=0.6)
        ax_sal_s.set_title("Speed Saliency Map"); ax_sal_s.axis('off')
        
        ax_cam_s.clear(); ax_cam_s.imshow(img_net.numpy()); ax_cam_s.imshow(cam_s, cmap='jet', alpha=0.5)
        ax_cam_s.set_title("Speed Grad-CAM"); ax_cam_s.axis('off')

        fig.canvas.draw_idle()

    def next_img(event):
        state['idx'] = (state['idx'] + 1) % len(results)
        update_plot()

    def prev_img(event):
        state['idx'] = (state['idx'] - 1) % len(results)
        update_plot()

    ax_prev = plt.axes([0.35, 0.02, 0.1, 0.05])
    btn_prev = Button(ax_prev, 'Previous Bad Pred', color='lightcoral')
    btn_prev.on_clicked(prev_img)

    ax_next = plt.axes([0.55, 0.02, 0.1, 0.05])
    btn_next = Button(ax_next, 'Next Bad Pred', color='lightgreen')
    btn_next.on_clicked(next_img)

    update_plot()
    plt.show()

if __name__ == "__main__":
    main()