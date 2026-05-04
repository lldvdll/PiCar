import os
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION (Set your best runs here)
# ==========================================
BEST_ANGLE_EXP = "41_V2S_keep_bot_angle"
BEST_SPEED_EXP = "40_V2S_keep_bot_speed" 

print(f"[INFO] Merging Angle from: {BEST_ANGLE_EXP}")
print(f"[INFO] Merging Speed from: {BEST_SPEED_EXP}")

# ==========================================
# 2. LOAD PREDICTIONS
# ==========================================
angle_path = os.path.join("experiments", BEST_ANGLE_EXP, "submission.csv")
speed_path = os.path.join("experiments", BEST_SPEED_EXP, "submission.csv")

angle_df = pd.read_csv(angle_path)
speed_df = pd.read_csv(speed_path)

# Verify alignment
if not angle_df['image_id'].equals(speed_df['image_id']):
    raise ValueError("ERROR: image_id columns do not match between the two CSVs!")

# ==========================================
# 3. BUILD FINAL DATAFRAME
# ==========================================
final_df = pd.DataFrame({
    'image_id': angle_df['image_id'],
    'angle': angle_df['angle'],
    'speed': speed_df['speed']
})

# ==========================================
# 4. SAFE ZONE SNAPPING (SPEED)
# ==========================================
print("[INFO] Applying Safe Zone Snapping to Speed...")
# Snap confident 'Go' predictions
final_df['speed'] = np.where(final_df['speed'] >= 0.75, 1.0, final_df['speed'])
# Snap confident 'Stop' predictions
final_df['speed'] = np.where(final_df['speed'] <= 0.25, 0.0, final_df['speed'])
# (Anything between 0.16 and 0.84 stays exactly as the raw probability)

# ==========================================
# 5. EXPORT
# ==========================================
output_file = "final_submission.csv"
final_df.to_csv(output_file, index=False)
print(f"[SUCCESS] Final Kaggle submission saved to: {output_file}")