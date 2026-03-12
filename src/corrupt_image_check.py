import os
import csv

img_dir = os.path.join("data", "training_data", "training_data")
csv_path = os.path.join("data", "bad_images.csv")

# Check if CSV exists so we know whether to write headers
file_exists = os.path.isfile(csv_path)

with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["filename", "reason"])
    
    count = 0
    for filename in os.listdir(img_dir):
        full_path = os.path.join(img_dir, filename)
        
        # If it's a file and exactly 0 bytes, flag it
        if os.path.isfile(full_path) and os.path.getsize(full_path) == 0:
            writer.writerow([filename, "Corrupt 0-byte file"])
            count += 1
            
print(f"[INFO] Found and appended {count} corrupt images to {csv_path}")