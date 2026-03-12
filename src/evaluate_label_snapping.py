
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from train_baseline_wandb import CONFIG, prepare_data_pipelines, preprocess_image, validate_image_paths

def snap_to_nearest(array, valid_values):
    differences = np.abs(array[:, np.newaxis] - valid_values)
    closest_indices = np.argmin(differences, axis=1)
    return valid_values[closest_indices]

def main():
    exp_dir = os.path.join("experiments", CONFIG["EXPERIMENT_NAME"])
    model_path = os.path.join(exp_dir, "best_model.h5")
    
    print(f"[INFO] Loading validation data and model for {CONFIG['EXPERIMENT_NAME']}...")
    _, val_ds = prepare_data_pipelines() 
    model = tf.keras.models.load_model(model_path)
    
    # ---------------------------------------------------------
    # PART 1: EVALUATE SQUARED ERROR DISTRIBUTIONS ON VAL SET
    # ---------------------------------------------------------
    print("[INFO] Extracting true labels and predicting on Validation Set...")
    y_true_angle, y_true_speed = [], []
    for images, labels in val_ds:
        y_true_angle.extend(labels['angle_output'].numpy())
        y_true_speed.extend(labels['speed_output'].numpy())
        
    y_true_angle = np.array(y_true_angle)
    y_true_speed = np.array(y_true_speed)
    
    # Generate predictions
    predictions = model.predict(val_ds)
    pred_raw_angle = predictions[0].flatten()
    pred_raw_speed = predictions[1].flatten()
    
    # Snap predictions
    valid_angles = np.linspace(0.0, 1.0, 17)
    pred_snap_angle = snap_to_nearest(pred_raw_angle, valid_angles)
    pred_snap_speed = np.where(pred_raw_speed >= 0.5, 1.0, 0.0)
    
    # Calculate SQUARED ERRORS for every single image
    se_raw_angle = (y_true_angle - pred_raw_angle) ** 2
    se_snap_angle = (y_true_angle - pred_snap_angle) ** 2
    se_raw_speed = (y_true_speed - pred_raw_speed) ** 2
    se_snap_speed = (y_true_speed - pred_snap_speed) ** 2
    
    print("\n=== OVERALL MSE RESULTS ===")
    print(f"Angle - Raw: {np.mean(se_raw_angle):.5f} | Snapped: {np.mean(se_snap_angle):.5f}")
    print(f"Speed - Raw: {np.mean(se_raw_speed):.5f} | Snapped: {np.mean(se_snap_speed):.5f}")
    
    # Plotting the ERROR distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # We use a log scale because SE usually heavily clusters near 0
    ax1.hist(se_raw_angle, bins=50, alpha=0.5, label='Raw Error', color='blue', log=True)
    ax1.hist(se_snap_angle, bins=50, alpha=0.5, label='Snapped Error', color='orange', log=True)
    ax1.set_title("Angle Squared Error Distribution (Lower is Better)")
    ax1.set_xlabel("Squared Error")
    ax1.set_ylabel("Number of Images (Log Scale)")
    ax1.legend()
    
    ax2.hist(se_raw_speed, bins=50, alpha=0.5, label='Raw Error', color='blue', log=True)
    ax2.hist(se_snap_speed, bins=50, alpha=0.5, label='Snapped Error', color='orange', log=True)
    ax2.set_title("Speed Squared Error Distribution (Lower is Better)")
    ax2.set_xlabel("Squared Error")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # PART 2: GENERATE SNAPPED SUBMISSION ON TEST SET
    # ---------------------------------------------------------
    print("\n[INFO] Generating snapped Kaggle submission on Test Set...")
    sub_df = pd.read_csv(CONFIG["SUBMISSION_TEMPLATE"])
    test_paths, sub_df = validate_image_paths(sub_df, CONFIG["TEST_IMG_DIR"])
    
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(CONFIG["BATCH_SIZE"])
    
    test_preds = model.predict(test_ds)
    test_raw_angle = test_preds[0].flatten()
    test_raw_speed = test_preds[1].flatten()
    
    sub_df['angle'] = snap_to_nearest(test_raw_angle, valid_angles)
    sub_df['speed'] = np.where(test_raw_speed >= 0.5, 1.0, 0.0)
    
    snapped_file = os.path.join(exp_dir, "submission_snapped.csv")
    sub_df.to_csv(snapped_file, index=False)
    print(f"[SUCCESS] Snapped submission saved to {snapped_file}")

if __name__ == "__main__":
    main()