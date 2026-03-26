import os
import pandas as pd
import tensorflow as tf

# 1. THE BULLETPROOF WORKAROUND: Define LayerScale ourselves so TF versions don't matter!
class LayerScale(tf.keras.layers.Layer):
    def __init__(self, init_values=1e-6, projection_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.projection_dim,) if self.projection_dim else (input_shape[-1],),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({
            "init_values": self.init_values,
            "projection_dim": self.projection_dim
        })
        return config

# 2. IMPORT PREPROCESSING FROM TRAINING SCRIPT
from train_bigger import CONFIG, validate_image_paths, read_and_decode_image

def main():
    exp_dir = os.path.join("experiments", CONFIG["EXPERIMENT_NAME"])
    model_path = os.path.join(exp_dir, "best_model.h5")
    output_csv = os.path.join(exp_dir, "submission.csv")

    print(f"[INFO] Loading model from: {model_path}")
    
    # 3. PASS OUR CUSTOM BLUEPRINT TO KERAS
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={"LayerScale": LayerScale}
    )

    print(f"[INFO] Loading Kaggle stencil from: {CONFIG['SUBMISSION_TEMPLATE']}")
    sub_df = pd.read_csv(CONFIG["SUBMISSION_TEMPLATE"])
    
    test_paths, sub_df = validate_image_paths(sub_df, CONFIG["TEST_IMG_DIR"])
    
    # BUILD DATASET
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(read_and_decode_image, num_parallel_calls=tf.data.AUTOTUNE).batch(CONFIG["BATCH_SIZE"])

    print("[INFO] Generating predictions...")
    preds = model.predict(test_ds, verbose=1)

    print("[INFO] Mapping raw predictions (No Snapping applied)...")
    submission_df = pd.DataFrame({
        'image_id': sub_df['image_id'],
        'angle': preds[0].flatten(),
        'speed': preds[1].flatten()
    })

    submission_df.to_csv(output_csv, index=False)
    print(f"======================================================")
    print(f"[SUCCESS] Kaggle submission saved to: {output_csv}")
    print(f"======================================================")

if __name__ == "__main__":
    main()