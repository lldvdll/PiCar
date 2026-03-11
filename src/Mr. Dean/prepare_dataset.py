import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_and_weight_data():
    csv_path = os.path.join("data", "train.csv")
    img_dir = os.path.join("data", "training_data", "training_data")
    output_csv = os.path.join("data", "train_clean_weighted.csv")
    
    print("[INFO] Loading raw training data...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['image_id'])
    
    # 1. Flag and remove broken/missing files permanently
    print("[INFO] Validating image files (this takes a moment)...")
    valid_paths, valid_indices = [], []
    for idx, row in df.iterrows():
        filename = str(int(float(row['image_id']))) + '.png'
        full_path = os.path.join(img_dir, filename).replace("\\", "/") 
        
        # Check if file exists and is not 0 bytes
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            valid_paths.append(full_path)
            valid_indices.append(idx)
            
    # Keep only rows with valid, existing images
    clean_df = df.loc[valid_indices].copy()
    clean_df['filepath'] = valid_paths # Save the exact path so training is faster!
    print(f"[INFO] Removed {len(df) - len(clean_df)} missing or broken images.")
    
    # 2. Discretize into Bins for Joint Frequency Calculation
    # We round to 3 decimal places to neatly group floating-point drift
    clean_df['angle_bin'] = clean_df['angle'].round(3)
    clean_df['speed_bin'] = clean_df['speed'].round(3)
    
    # 3. Calculate Joint Frequencies & Assign Weights
    print("[INFO] Calculating joint probability weights...")
    # Count how many times each (angle, speed) pair occurs
    clean_df['pair_freq'] = clean_df.groupby(['angle_bin', 'speed_bin'])['angle_bin'].transform('count')
    
    # Assign Weight: 1 / frequency (Rare pairs get high weights, common pairs get tiny weights)
    clean_df['sample_weight'] = 1.0 / clean_df['pair_freq']
    
    # Save the pristine dataset
    clean_df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Saved clean, weighted dataset to: {output_csv}")
    
    # 4. Generate the Heatmap Visualization
    joint_dist = pd.crosstab(clean_df['angle_bin'], clean_df['speed_bin'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(joint_dist, annot=True, fmt='d', cmap='YlOrRd')
    plt.title("Joint Distribution of Steering Angle and Speed (Training Data)")
    plt.xlabel("Speed")
    plt.ylabel("Steering Angle")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    prepare_and_weight_data()