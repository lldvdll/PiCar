import os
import pandas as pd
import tensorflow as tf
import numpy as np
from train_baseline_wandb import CONFIG, preprocess_image, validate_image_paths

def main():
    exp_dir = os.path.join("experiments", CONFIG["EXPERIMENT_NAME"])
    model_path = os.path.join(exp_dir, "best_model.h5")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] No trained model found at {model_path}")
        
    print(f"[INFO] Loading model from {exp_dir}...")
    best_model = tf.keras.models.load_model(model_path)
    
    print("[INFO] Preparing Kaggle test data...")
    sub_df = pd.read_csv(CONFIG["SUBMISSION_TEMPLATE"])
    test_paths, sub_df = validate_image_paths(sub_df, CONFIG["TEST_IMG_DIR"])
    
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(CONFIG["BATCH_SIZE"])
    
    print("[INFO] Generating predictions...")
    predictions = best_model.predict(test_ds)
    
    sub_df['angle'] = predictions[0].flatten()
    sub_df['speed'] = predictions[1].flatten()
    
    submission_file = os.path.join(exp_dir, "submission.csv")
    sub_df.to_csv(submission_file, index=False)
    print(f"[SUCCESS] Submission saved to {submission_file}")

if __name__ == "__main__":
    main()