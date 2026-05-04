import numpy as np
import tensorflow as tf
import os

class Model:
    # Point this to the name of your saved weights file
    saved_model_path = 'best_model.h5'

    def __init__(self):
        # Load your single multi-head model
        model_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_model_path)
        self.model = tf.keras.models.load_model(model_filepath)
        print("[INFO] Multi-head model loaded successfully.")

    def preprocess(self, image):
        # 1. Cast to float and normalize to 0.0 - 1.0 (Exact match to training)
        im = tf.cast(image, tf.float32) / 255.0
        
        # 2. Crop top 110 pixels and bottom 30 pixels (Exact match to training)
        im = im[110:-30, :, :]
        
        # 3. Resize to the target height/width (96x160)
        im = tf.image.resize(im, [96, 160])
        
        # 4. Add the batch dimension (so shape becomes 1 x 96 x 160 x 3)
        im = tf.expand_dims(im, axis=0)
        
        return im

    def predict(self, image):
        # Preprocess the raw camera frame
        image = self.preprocess(image)
        
        # Run inference. Because of your multi-head architecture, 
        # preds is a list of two arrays: [angle_array, speed_array]
        preds = self.model.predict(image, verbose=0)
        
        # Extract the raw 0.0 to 1.0 float values
        pred_angle_raw = float(preds[0][0][0])
        pred_speed_raw = float(preds[1][0][0])
        
        # --- SPEED CONVERSION ---
        # Snap the 0-1 probability to a hard 0 or 1, then multiply by hardware speed (35)
        speed = int(np.round(pred_speed_raw)) * 35
        
        # --- ANGLE CONVERSION ---
        # The example code mapped 17 classes to physical angles between 50 and 130.
        # Your model outputs a continuous 0.0 to 1.0. We map that linearly to 50-130.
        # (e.g., 0.0 = 50, 0.5 = 90 straight ahead, 1.0 = 130)
        angle = (pred_angle_raw * 80.0) + 50.0
        
        print(f'Raw Angle: {pred_angle_raw:.3f} -> Physical Angle: {angle:.1f} | Speed: {speed}')
        
        return angle, speed