import os
import random
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tensorflow as tf

# Import the exact configuration and functions from your live training script!
from train_baseline_wandb import CONFIG, preprocess_image, augment_image

def load_random_image_data(csv_path, img_dir, sample_size=100):
    print(f"[INFO] Loading images from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    valid_data = []
    for _, row in df.dropna(subset=['image_id']).iterrows():
        filename = str(int(float(row['image_id']))) + '.png'
        full_path = os.path.join(img_dir, filename).replace("\\", "/")
        
        if os.path.exists(full_path):
            valid_data.append({
                'filepath': full_path,
                'filename': filename,
                'angle': row.get('angle', 'N/A'),
                'speed': row.get('speed', 'N/A')
            })
                
    random.shuffle(valid_data)
    return valid_data[:sample_size]

def main():
    image_data = load_random_image_data(CONFIG["TRAIN_CSV"], CONFIG["TRAIN_IMG_DIR"])
    if not image_data:
        raise ValueError("No images found! Check your CONFIG paths.")

    bad_images_csv = "bad_images.csv"
    if not os.path.exists(bad_images_csv):
        with open(bad_images_csv, "w", newline="") as f:
            csv.writer(f).writerow(["filename", "reason"])

    # Setup the 3-panel Matplotlib figure
    fig, (ax_raw, ax_proc, ax_aug) = plt.subplots(1, 3, figsize=(16, 6))
    plt.subplots_adjust(bottom=0.25, left=0.2) # Shift left to make room for the legend
    
    # Add a text box to the left side showing the current Augmentation Config
    legend_text = (
        "Augmentation Config:\n"
        f"Brightness: {CONFIG['AUG_BRIGHTNESS_DELTA']}\n"
        f"Contrast: {CONFIG['AUG_CONTRAST_LOWER']} - {CONFIG['AUG_CONTRAST_UPPER']}\n"
        f"Saturation: {CONFIG['AUG_SATURATION_LOWER']} - {CONFIG['AUG_SATURATION_UPPER']}\n"
        f"Hue Delta: {CONFIG['AUG_HUE_DELTA']}\n"
        f"Noise StdDev: {CONFIG['AUG_NOISE_STDDEV']}\n"
        f"Rotation: {CONFIG['AUG_ROTATION_FACTOR']}"
    )
    fig.text(0.02, 0.5, legend_text, fontsize=10, va='center', bbox=dict(facecolor='white', alpha=0.8))

    state = {'current_idx': 0, 'processed_tensor': None}
    
    def update_plot(new_image=True):
        if state['current_idx'] >= len(image_data):
            print("[INFO] Reached the end of the sampled images.")
            return
            
        item = image_data[state['current_idx']]
        
        # If we clicked "Next Image", load and preprocess it
        if new_image:
            raw_tensor = tf.io.read_file(item['filepath'])
            raw_tensor = tf.image.decode_png(raw_tensor, channels=CONFIG["CHANNELS"])
            
            # Get the strictly cropped/resized version (augment=False)
            state['processed_tensor'] = preprocess_image(item['filepath'], augment=False)
            
            ax_raw.clear()
            ax_raw.imshow(raw_tensor.numpy())
            ax_raw.set_title(f"{item['filename']} | Angle: {item['angle']}\nRaw Shape: {raw_tensor.shape}")
            ax_raw.axis('off')
            
            ax_proc.clear()
            ax_proc.imshow(state['processed_tensor'].numpy()) 
            ax_proc.set_title(f"Preprocessed (Base)\nShape: {state['processed_tensor'].shape}")
            ax_proc.axis('off')

        # EVERY time this runs (even on Re-Augment), generate a fresh random augmentation
        augmented_tensor = augment_image(state['processed_tensor'])
        
        ax_aug.clear()
        ax_aug.imshow(augmented_tensor.numpy())
        ax_aug.set_title("Live Random Augmentation")
        ax_aug.axis('off')
        
        fig.canvas.draw()

    def next_image(event=None):
        state['current_idx'] += 1
        update_plot(new_image=True)
        
    def re_augment(event=None):
        # Pass False so it re-uses the base image but rolls new random augmentation math
        update_plot(new_image=False) 
        
    def flag_image(event=None):
        item = image_data[state['current_idx']]
        with open(bad_images_csv, "a", newline="") as f:
            csv.writer(f).writerow([item['filename'], "Manually flagged in visualizer"])
        print(f"[INFO] Flagged {item['filename']} as bad.")
        next_image()

    # Create the buttons
    ax_btn_reaug = plt.axes([0.3, 0.05, 0.15, 0.075])
    btn_reaug = Button(ax_btn_reaug, 'Re-Augment Image', color='lightblue')
    btn_reaug.on_clicked(re_augment)
    
    ax_btn_flag = plt.axes([0.5, 0.05, 0.15, 0.075])
    btn_flag = Button(ax_btn_flag, 'Flag as Bad', color='salmon')
    btn_flag.on_clicked(flag_image)
    
    ax_btn_next = plt.axes([0.7, 0.05, 0.15, 0.075])
    btn_next = Button(ax_btn_next, 'Next Image')
    btn_next.on_clicked(next_image)

    # Show the first image immediately
    update_plot(new_image=True)
    plt.show()

if __name__ == "__main__":
    main()