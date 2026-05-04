import os
import hashlib
import pandas as pd
from PIL import Image

# Use the exact paths from your configs
TRAIN_DIR = os.path.join("data", "training_data", "training_data")
TEST_DIR = os.path.join("data", "test_data", "test_data")
OUTPUT_CSV = "data_leakage_report.csv"

def get_pixel_hash(filepath):
    """
    Opens an image, strips all metadata, and returns an MD5 hash 
    of the raw pixel data.
    """
    try:
        with Image.open(filepath) as img:
            # Convert to standard RGB to ensure mathematically identical 
            # bytes regardless of how the PNG was originally saved
            img = img.convert("RGB")
            pixel_data = img.tobytes()
            return hashlib.md5(pixel_data).hexdigest()
    except Exception as e:
        print(f"[ERROR] Could not process {filepath}: {e}")
        return None

def find_duplicates():
    seen_hashes = {}
    results = []

    def process_directory(directory, split_name):
        if not os.path.exists(directory):
            print(f"[WARNING] Directory not found: {directory}")
            return

        print(f"[INFO] Scanning {split_name} directory: {directory}")
        
        # Gather all valid images
        valid_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in valid_files:
            filepath = os.path.join(directory, filename).replace("\\", "/")
            img_hash = get_pixel_hash(filepath)

            if img_hash is None:
                continue

            # Check for collision
            if img_hash in seen_hashes:
                results.append({
                    "image_location": filepath,
                    "split": split_name,
                    "is_duplicate": True,
                    "duplicate_of": seen_hashes[img_hash]
                })
            else:
                # First time seeing this exact pixel arrangement
                seen_hashes[img_hash] = filepath
                results.append({
                    "image_location": filepath,
                    "split": split_name,
                    "is_duplicate": False,
                    "duplicate_of": None
                })

    # Process Train first, so if a Test image matches, it flags as a duplicate of Train
    process_directory(TRAIN_DIR, "train")
    process_directory(TEST_DIR, "test")

    # Generate Report
    print("\n[INFO] Saving results...")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Print Diagnostics
    total_images = len(df)
    total_dupes = len(df[df['is_duplicate'] == True])
    test_leakage = len(df[(df['is_duplicate'] == True) & (df['split'] == 'test')])
    
    print("\n" + "="*50)
    print(" 🚨 DATA LEAKAGE REPORT 🚨")
    print("="*50)
    print(f"Total Images Scanned : {total_images}")
    print(f"Total Duplicates     : {total_dupes}")
    print(f"Test Images Leaked   : {test_leakage}")
    print("="*50)
    print(f"Full detailed report saved to: {OUTPUT_CSV}\n")

if __name__ == "__main__":
    find_duplicates()