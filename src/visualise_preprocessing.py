import os
import random
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tensorflow as tf

# Import the exact configuration and function directly from your training script!
from train_baseline_wandb import CONFIG, preprocess_image

def load_random_image_data(csv_path, img_dir, sample_size=100):
    """Loads a pool of random image metadata from your dataset."""
    print(f"[INFO] Loading images from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    valid_data = []
    for _, row in df.dropna(subset=['image_id']).iterrows():
        # Handle the trailing .0 if pandas read the ID as a float
        filename = str(int(float(row['image_id']))) + '.png'
        full_path = os.path.join(img_dir, filename).replace("\\", "/")
        
        if os.path.exists(full_path):
            valid_data.append({
                'filepath': full_path,
                'filename': filename,
                'angle': row.get('angle', 'N/A'),
                'speed': row.get('speed', 'N/A')
            })
                
    # Shuffle and grab a subset so it's fast
    random.shuffle(valid_data)
    return valid_data[:sample_size]

def main():
    image_data = load_random_image_data(CONFIG["TRAIN_CSV"], CONFIG["TRAIN_IMG_DIR"])
    if not image_data:
        raise ValueError("No images found! Check your CONFIG paths.")

    # Ensure the bad images CSV has headers if it doesn't exist
    bad_images_csv = os.path.join("data", "bad_images.csv")
    if not os.path.exists(bad_images_csv):
        with open(bad_images_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "reason"])

    # Setup the Matplotlib figure
    fig, (ax_raw, ax_proc) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25) # Make room for the buttons at the bottom
    
    # State dictionary to keep track of the current image index
    state = {'current_idx': 0}
    
    def update_plot():
        if state['current_idx'] >= len(image_data):
            print("[INFO] Reached the end of the sampled images.")
            return
            
        item = image_data[state['current_idx']]
        
        # 1. Load the RAW image directly to see the true "Before"
        raw_tensor = tf.io.read_file(item['filepath'])
        raw_tensor = tf.image.decode_png(raw_tensor, channels=CONFIG["CHANNELS"])
        
        # 2. Run YOUR preprocessing function from the training script for the "After"
        # (Since we only pass the path, it returns just the image tensor, not the labels)
        processed_tensor = preprocess_image(item['filepath'])
        
        # 3. Update the Left Plot (Raw)
        ax_raw.clear()
        ax_raw.imshow(raw_tensor.numpy())
        title_text = f"File: {item['filename']} | Angle: {item['angle']} | Speed: {item['speed']}\nRaw Shape: {raw_tensor.shape}"
        ax_raw.set_title(title_text)
        ax_raw.axis('off')
        
        # 4. Update the Right Plot (Preprocessed)
        ax_proc.clear()
        ax_proc.imshow(processed_tensor.numpy()) 
        ax_proc.set_title(f"Preprocessed (Crop Top: {CONFIG.get('CROP_TOP_PIXELS', 0)}px)\nShape: {processed_tensor.shape}")
        ax_proc.axis('off')
        
        fig.canvas.draw()

    def next_image(event=None):
        state['current_idx'] += 1
        update_plot()
        
    def flag_image(event=None):
        item = image_data[state['current_idx']]
        
        # Append to our manual exclusion list
        with open(bad_images_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([item['filename'], "Manually flagged in visualizer"])
            
        print(f"[INFO] Flagged {item['filename']} as bad. Saved to {bad_images_csv}")
        next_image() # Automatically advance after flagging so workflow is fast

    # Create the buttons
    ax_btn_next = plt.axes([0.55, 0.05, 0.15, 0.075])
    btn_next = Button(ax_btn_next, 'Next Image')
    btn_next.on_clicked(next_image)
    
    ax_btn_flag = plt.axes([0.30, 0.05, 0.15, 0.075])
    btn_flag = Button(ax_btn_flag, 'Flag as Bad', color='salmon', hovercolor='red')
    btn_flag.on_clicked(flag_image)

    # Show the first image immediately
    update_plot()
    
    print("[INFO] Close the window to exit the visualizer.")
    plt.show()

if __name__ == "__main__":
    main()