import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

from train_baseline_wandb import CONFIG, preprocess_image

def load_random_image_data(csv_path, img_dir, sample_size=50):
    print(f"[INFO] Loading images from {csv_path}...")
    df = pd.read_csv(csv_path)
    valid_data = []
    for _, row in df.dropna(subset=['image_id']).iterrows():
        filename = str(int(float(row['image_id']))) + '.png'
        full_path = os.path.join(img_dir, filename).replace("\\", "/")
        if os.path.exists(full_path):
            valid_data.append({'filepath': full_path, 'angle': row.get('angle', 'N/A'), 'speed': row.get('speed', 'N/A')})
    random.shuffle(valid_data)
    return valid_data[:sample_size]

def make_dual_gradcam(img_array, model, base_model, last_conv_layer_name):
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    base_grad_model = tf.keras.Model(inputs=base_model.inputs, outputs=[last_conv_layer.output, base_model.output])
    
    head_input = tf.keras.Input(shape=base_model.output_shape[1:])
    x = head_input
    base_model_index = model.layers.index(base_model)
    
    for layer in model.layers[base_model_index + 1:]:
        if layer.name in ['angle_output', 'speed_output']:
            continue
        x = layer(x)
        
    angle_out = model.get_layer('angle_output')(x)
    speed_out = model.get_layer('speed_output')(x)
    head_model = tf.keras.Model(inputs=head_input, outputs=[angle_out, speed_out])

    with tf.GradientTape(persistent=True) as tape:
        conv_outputs, base_outputs = base_grad_model(img_array)
        tape.watch(conv_outputs) 
        
        preds = head_model(base_outputs)
        angle_pred = preds[0][:, 0]
        speed_pred = preds[1][:, 0]

    grads_angle = tape.gradient(angle_pred, conv_outputs)
    pooled_grads_angle = tf.reduce_mean(grads_angle, axis=(0, 1, 2))
    heatmap_angle = conv_outputs[0] @ pooled_grads_angle[..., tf.newaxis]
    heatmap_angle = tf.maximum(tf.squeeze(heatmap_angle), 0)
    heatmap_angle /= (tf.math.reduce_max(heatmap_angle) + 1e-10)

    grads_speed = tape.gradient(speed_pred, conv_outputs)
    pooled_grads_speed = tf.reduce_mean(grads_speed, axis=(0, 1, 2))
    heatmap_speed = conv_outputs[0] @ pooled_grads_speed[..., tf.newaxis]
    heatmap_speed = tf.maximum(tf.squeeze(heatmap_speed), 0)
    heatmap_speed /= (tf.math.reduce_max(heatmap_speed) + 1e-10)

    del tape
    return heatmap_angle.numpy(), heatmap_speed.numpy()

def main():
    model_path = os.path.join("experiments", "05_agressive_image_crop", "best_model.h5")
    print("[INFO] Loading Model...")
    model = tf.keras.models.load_model(model_path)
    base_model = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)][0]
    
    layers_to_inspect = [l.name for l in base_model.layers if 'expand_relu' in l.name] + ['out_relu']
    
    image_data = load_random_image_data(CONFIG["TRAIN_CSV"], CONFIG["TRAIN_IMG_DIR"])

    fig = plt.figure(figsize=(18, 9))
    gs_master = gridspec.GridSpec(1, 4, figure=fig)
    
    gs_left = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[0, 0], hspace=0.3)
    ax_raw = fig.add_subplot(gs_left[0, 0])
    ax_cam_angle = fig.add_subplot(gs_left[1, 0])
    ax_cam_speed = fig.add_subplot(gs_left[2, 0])

    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.98, top=0.90)

    state = {
        'current_idx': 0, 'layer_idx': 0, 'feature_page': 0, 
        'img_tensor': None, 'features': None, 'ax_feats': [], 'features_per_page': 16
    }

    info_text = fig.text(0.30, 0.95, "", fontsize=12, fontweight='bold')

    def update_plot(new_image=False, new_layer=False):
        item = image_data[state['current_idx']]
        layer_name = layers_to_inspect[state['layer_idx']]
        
        if new_image:
            img_tensor = preprocess_image(item['filepath'], augment=False)
            state['img_tensor'] = tf.expand_dims(img_tensor, 0)
            
            preds = model.predict(state['img_tensor'], verbose=0)
            pred_angle = preds[0][0][0]
            pred_speed = preds[1][0][0]
            
            ax_raw.clear()
            ax_raw.imshow(img_tensor.numpy())
            ax_raw.set_title(f"True A: {item['angle']} | Pred A: {pred_angle:.2f}\nTrue S: {item['speed']} | Pred S: {pred_speed:.2f}", fontsize=10)
            ax_raw.axis('off')
            
            hm_angle, hm_speed = make_dual_gradcam(state['img_tensor'], model, base_model, layer_name)
            
            hm_angle_rs = tf.image.resize(hm_angle[..., tf.newaxis], [CONFIG["IMG_HEIGHT_TARGET"], CONFIG["IMG_WIDTH_TARGET"]])
            ax_cam_angle.clear()
            ax_cam_angle.imshow(img_tensor.numpy())
            ax_cam_angle.imshow(hm_angle_rs[:, :, 0], cmap='jet', alpha=0.5) 
            ax_cam_angle.set_title("Grad-CAM: Angle Focus", fontsize=10)
            ax_cam_angle.axis('off')

            hm_speed_rs = tf.image.resize(hm_speed[..., tf.newaxis], [CONFIG["IMG_HEIGHT_TARGET"], CONFIG["IMG_WIDTH_TARGET"]])
            ax_cam_speed.clear()
            ax_cam_speed.imshow(img_tensor.numpy())
            ax_cam_speed.imshow(hm_speed_rs[:, :, 0], cmap='jet', alpha=0.5) 
            ax_cam_speed.set_title("Grad-CAM: Speed Focus", fontsize=10)
            ax_cam_speed.axis('off')

        if new_image or new_layer:
            feature_extractor = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.get_layer(layer_name).output)
            state['features'] = feature_extractor.predict(state['img_tensor'], verbose=0)[0]
            state['feature_page'] = 0 
            
            for ax in state['ax_feats']:
                ax.remove()
            state['ax_feats'] = []
            
            fm_h, fm_w = state['features'].shape[0], state['features'].shape[1]
            cols = min(12, max(4, int(CONFIG["IMG_WIDTH_TARGET"] // fm_w)))
            rows = min(10, max(4, int(CONFIG["IMG_HEIGHT_TARGET"] // fm_h)))
            
            state['features_per_page'] = rows * cols
            
            # Tighter spacing: wspace=0.05, hspace=0.05
            gs_right = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs_master[0, 1:], wspace=0.05, hspace=0.05)
            for i in range(rows):
                for j in range(cols):
                    state['ax_feats'].append(fig.add_subplot(gs_right[i, j]))
            
        total_features = state['features'].shape[-1]
        start_idx = state['feature_page'] * state['features_per_page']
        end_idx = min(start_idx + state['features_per_page'], total_features)
        
        info_text.set_text(f"Layer: {layer_name} (Depth: {state['layer_idx']+1}/{len(layers_to_inspect)})\nViewing Features: {start_idx} to {end_idx-1} (Total: {total_features})")

        for i, ax in enumerate(state['ax_feats']):
            ax.clear()
            feat_idx = start_idx + i
            if feat_idx < total_features:
                ax.imshow(state['features'][:, :, feat_idx], cmap='viridis')
            ax.axis('off')
            
        fig.canvas.draw()

    def next_image(event=None):
        state['current_idx'] = (state['current_idx'] + 1) % len(image_data)
        update_plot(new_image=True)
        
    def next_layer(event=None):
        state['layer_idx'] = (state['layer_idx'] + 1) % len(layers_to_inspect)
        update_plot(new_image=False, new_layer=True)

    def prev_layer(event=None):
        state['layer_idx'] = (state['layer_idx'] - 1) % len(layers_to_inspect)
        update_plot(new_image=False, new_layer=True)
        
    def next_feature_page(event=None):
        total_features = state['features'].shape[-1]
        state['feature_page'] += 1
        if state['feature_page'] * state['features_per_page'] >= total_features:
            state['feature_page'] = 0 
        update_plot(new_image=False, new_layer=False)

    def prev_feature_page(event=None):
        total_features = state['features'].shape[-1]
        max_page = (total_features - 1) // state['features_per_page']
        state['feature_page'] -= 1
        if state['feature_page'] < 0:
            state['feature_page'] = max_page
        update_plot(new_image=False, new_layer=False)

    # 5 Buttons layout
    ax_btn_prev_layer = plt.axes([0.15, 0.02, 0.12, 0.05])
    btn_prev_layer = Button(ax_btn_prev_layer, 'Prev Layer', color='lightblue')
    btn_prev_layer.on_clicked(prev_layer)

    ax_btn_next_layer = plt.axes([0.28, 0.02, 0.12, 0.05])
    btn_next_layer = Button(ax_btn_next_layer, 'Next Layer', color='lightblue')
    btn_next_layer.on_clicked(next_layer)
    
    ax_btn_prev_feat = plt.axes([0.45, 0.02, 0.12, 0.05])
    btn_prev_feat = Button(ax_btn_prev_feat, 'Prev Page', color='lightgreen')
    btn_prev_feat.on_clicked(prev_feature_page)

    ax_btn_next_feat = plt.axes([0.58, 0.02, 0.12, 0.05])
    btn_next_feat = Button(ax_btn_next_feat, 'Next Page', color='lightgreen')
    btn_next_feat.on_clicked(next_feature_page)
    
    ax_btn_next_img = plt.axes([0.75, 0.02, 0.12, 0.05])
    btn_next_img = Button(ax_btn_next_img, 'Next Image', color='lightgray')
    btn_next_img.on_clicked(next_image)

    update_plot(new_image=True)
    plt.show()

if __name__ == "__main__":
    main()