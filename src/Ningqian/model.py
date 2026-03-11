"""
Two options:
1. NVIDIA CNN — custom CNN based on NVIDIA's self-driving paper (Bojarski et al., 2016)
2. MobileNetV2 — transfer learning with two-phase training

Both output 2 values: [angle, speed] normalised to [0, 1].
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_nvidia_model(input_shape=(87, 200, 3)):
    """
    NVIDIA's End-to-End Self-Driving CNN (modified for angle + speed).

    Architecture from the paper:
    - 5 conv layers with increasing filters (24, 36, 48, 64, 64)
    - 3 dense layers (100, 50, 10)
    - Output: 2 neurons with sigmoid (angle, speed in [0,1])

    Added BatchNorm + Dropout for better generalisation.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Conv block 1-2: 5x5 filters with stride 2 (downsampling)
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        layers.BatchNormalization(),

        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        layers.BatchNormalization(),

        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        layers.BatchNormalization(),

        # Conv block 3: 3x3 filters, no stride
        layers.Conv2D(64, (3, 3), activation='elu'),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='elu'),
        layers.BatchNormalization(),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.3),

        layers.Dense(100, activation='elu'),
        layers.Dropout(0.3),

        layers.Dense(50, activation='elu'),
        layers.Dropout(0.2),

        layers.Dense(10, activation='elu'),

        # Output: angle and speed (both normalised 0-1)
        layers.Dense(2, activation='sigmoid'),
    ])

    return model


def build_mobilenet_model(input_shape=(87, 200, 3), freeze_base=True):
    """
    MobileNetV2 transfer learning model.

    Uses ImageNet pre-trained weights. The base can be frozen (for warmup phase)
    then unfrozen for fine-tuning (two-phase training).

    Returns: (model, base_model) — base_model handle needed for unfreezing later.
    """
    inputs = layers.Input(shape=input_shape)

    # MobileNetV2 needs at least 96x96 input — resize internally
    x = layers.Resizing(96, 96)(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = not freeze_base

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='elu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model, base_model


if __name__ == '__main__':
    print("=" * 60)
    print("NVIDIA Model:")
    print("=" * 60)
    nvidia = build_nvidia_model()
    nvidia.summary()

    print("\n" + "=" * 60)
    print("MobileNetV2 Model:")
    print("=" * 60)
    mobile, base = build_mobilenet_model()
    mobile.summary()
    print(f"\nBase model layers: {len(base.layers)}")
    print(f"Base model trainable: {base.trainable}")
