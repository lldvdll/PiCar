"""
Loads a trained model, predicts on test images, writes Kaggle submission CSV.

Usage:
    python src/Ningqian/predict.py --exp_dir experiments/my_experiment --model nvidia
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd
import tensorflow as tf
from Ningqian.dataset import create_test_dataset


def predict(args):
    # ============================================================
    # 1. Load model
    # ============================================================
    model_path = os.path.join(args.exp_dir, f'best_{args.model}.keras')
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Run training first: python src/rocky/train.py")
        return

    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully")

    # ============================================================
    # 2. Load test data
    # ============================================================
    test_ds, image_ids = create_test_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # ============================================================
    # 3. Generate predictions
    # ============================================================
    print("Generating predictions...")
    predictions = model.predict(test_ds)
    print(f"Predictions shape: {predictions.shape}")

    angles = predictions[:, 0].clip(0, 1)
    speeds = predictions[:, 1].clip(0, 1)

    # ============================================================
    # 4. Create submission CSV
    # ============================================================
    submission = pd.DataFrame({
        'image_id': image_ids,
        'angle': angles,
        'speed': speeds,
    })
    submission = submission.sort_values('image_id').reset_index(drop=True)

    output_path = os.path.join(args.exp_dir, 'submission.csv')
    submission.to_csv(output_path, index=False)

    print(f"\nSubmission saved to: {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"\nFirst 5 rows:")
    print(submission.head())
    print(f"\nAngle stats: mean={angles.mean():.4f}, std={angles.std():.4f}")
    print(f"Speed stats: mean={speeds.mean():.4f}, std={speeds.std():.4f}")

    # ============================================================
    # 5. Verify format
    # ============================================================
    sample_path = os.path.join(args.data_dir, 'sample_submission.csv')
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        if submission.shape == sample.shape:
            print("\n[OK] Shape matches sample submission!")
        else:
            print(f"\n[WARNING] Shape mismatch! Ours: {submission.shape}, Expected: {sample.shape}")

    print(f"\nTo submit:")
    print(f"  kaggle competitions submit -c machine-learning-in-science-ii-2026 "
          f"-f {output_path} -m '{args.model} submission'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Kaggle submission')
    parser.add_argument('--model', type=str, default='nvidia',
                        choices=['nvidia', 'mobilenet'],
                        help='Model architecture (default: nvidia)')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Experiment directory containing the saved model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory (default: data)')
    args = parser.parse_args()

    predict(args)
