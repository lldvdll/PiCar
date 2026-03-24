import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_training_data():
    csv_path = os.path.join("data", "train.csv")
    bad_csv_path = os.path.join("data", "bad_images.csv")
    
    print(f"[INFO] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Load bad_images
    bad_df = pd.read_csv(bad_csv_path)
    bad_list = bad_df['filename'].astype(str).tolist()
    df['check_name'] = df['image_id'].astype(float).astype(int).astype(str) + '.png'
    initial_count = len(df)
    df = df[~df['check_name'].isin(bad_list)].drop(columns=['check_name'])
    print(f"[INFO] Dropped {initial_count - len(df)} manually flagged bad images.")
    
    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Angle Distribution
    ax1.hist(df['angle'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution of Steering Angles')
    ax1.set_xlabel('Angle (Normalized 0 to 1)')
    ax1.set_ylabel('Number of Images')
    
    # Plot Speed Distribution
    ax2.hist(df['speed'], bins=50, color='green', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution of Speeds')
    ax2.set_xlabel('Speed (Normalized 0 to 1)')
    ax2.set_ylabel('Number of Images')
    
    plt.tight_layout()
    print("[INFO] Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    analyze_training_data()