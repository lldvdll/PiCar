"""
Key features combined from both Rocky and David's approaches:
- Weighted MSE loss (70% angle, 30% speed) — Rocky's idea
- Inverse-frequency data balancing — David's idea
- Two-phase training (warmup + fine-tune) — David's idea
- Contiguous block validation split — Rocky's idea
- Data augmentation (flip, brightness, shadow) — Rocky's idea
- Weights & Biases experiment tracking — shared

Usage:
    python src/Ningqian/train.py                          # NVIDIA CNN (default)
    python src/Ningqian/train.py --model mobilenet        # MobileNetV2 with two-phase training
    python src/Ningqian/train.py --no_wandb               # Skip W&B logging
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import json
import numpy as np
import tensorflow as tf
from Ningqian.dataset import create_datasets, create_test_dataset
from Ningqian.model import build_nvidia_model, build_mobilenet_model


# ============================================================
# W&B Configuration (shared with David)
# ============================================================
WANDB_PROJECT = "PiCar"
WANDB_ENTITY = "lpxdv2-university-of-nottingham"


def weighted_mse(y_true, y_pred):
    """
    Weighted MSE loss: 70% angle + 30% speed.

    Angle prediction is harder (17 discrete values spread across [0,1])
    while speed is binary (0 or 1), so we weight angle more.
    """
    angle_loss = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))
    speed_loss = tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1]))
    return 0.7 * angle_loss + 0.3 * speed_loss


def train(args):
    """Main training function."""

    # ============================================================
    # 1. Setup experiment directory
    # ============================================================
    exp_name = args.experiment_name or f"rocky_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join('experiments', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config for reproducibility
    config = vars(args)
    config['experiment_name'] = exp_name
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'=' * 60}")

    # ============================================================
    # 2. Initialise W&B (optional)
    # ============================================================
    wandb_run = None
    callbacks = []

    if not args.no_wandb:
        try:
            import wandb
            from wandb.integration.keras import WandbMetricsLogger

            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=exp_name,
                dir=exp_dir,
                config=config,
                notes=args.description or f"Rocky's {args.model} model",
            )
            callbacks.append(WandbMetricsLogger())
            print("[INFO] W&B logging enabled")
        except ImportError:
            print("[WARNING] wandb not installed — skipping experiment tracking")
            print("         Install with: pip install wandb")
        except Exception as e:
            print(f"[WARNING] W&B init failed: {e}")
            print("         Training will continue without W&B logging")

    # ============================================================
    # 3. Load data
    # ============================================================
    print("\nLoading data...")
    train_ds, val_ds, n_train, n_val = create_datasets(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        augment=True,
        balance_data=True,  # Inverse-frequency balancing (David's approach)
    )

    # ============================================================
    # 4. Build model
    # ============================================================
    print(f"\nBuilding {args.model} model...")

    if args.model == 'nvidia':
        model = build_nvidia_model()
        base_model = None  # No two-phase for NVIDIA (trained from scratch)
    elif args.model == 'mobilenet':
        model, base_model = build_mobilenet_model(freeze_base=True)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=weighted_mse,
        metrics=['mae'],
    )
    model.summary()

    # ============================================================
    # 5. Callbacks
    # ============================================================
    save_path = os.path.join(exp_dir, f'best_{args.model}.keras')

    callbacks.extend([
        tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
    ])

    # ============================================================
    # 6. Train — Phase 1 (Warmup)
    # ============================================================
    start_time = time.time()
    print(f"\nTraining samples: {n_train}, Validation samples: {n_val}")

    if args.model == 'mobilenet' and args.warmup_epochs > 0:
        # --- TWO-PHASE TRAINING (David's approach) ---
        print(f"\n--- PHASE 1: WARMUP ({args.warmup_epochs} epochs, frozen backbone) ---")
        history_1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.warmup_epochs,
            callbacks=callbacks,
        )
        all_val_loss = list(history_1.history['val_loss'])

        # --- PHASE 2: Fine-tune ---
        if args.finetune_epochs > 0 and args.unfreeze_layers > 0:
            print(f"\n--- PHASE 2: FINE-TUNING top {args.unfreeze_layers} layers "
                  f"({args.finetune_epochs} epochs) ---")

            # Unfreeze the top N layers of the backbone
            base_model.trainable = True
            for layer in base_model.layers[:-args.unfreeze_layers]:
                layer.trainable = False

            # Recompile with lower learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.finetune_lr),
                loss=weighted_mse,
                metrics=['mae'],
            )

            total_epochs = args.warmup_epochs + args.finetune_epochs
            history_2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=total_epochs,
                initial_epoch=args.warmup_epochs,  # Continue from where warmup ended
                callbacks=callbacks,
            )
            all_val_loss.extend(history_2.history['val_loss'])
    else:
        # --- SINGLE-PHASE TRAINING (for NVIDIA CNN) ---
        print(f"\nTraining for up to {args.epochs} epochs...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
        )
        all_val_loss = list(history.history['val_loss'])

    # ============================================================
    # 7. Results
    # ============================================================
    training_time = (time.time() - start_time) / 60
    best_epoch = int(np.argmin(all_val_loss)) + 1
    best_val_loss = float(min(all_val_loss))

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Training time:      {training_time:.1f} minutes")
    print(f"Best epoch:         {best_epoch}")
    print(f"Best val MSE:       {best_val_loss:.6f}")
    print(f"Estimated mark:     {100 - 800 * best_val_loss:.1f}%")
    print(f"Model saved to:     {save_path}")

    # Log to W&B
    if wandb_run is not None:
        import wandb
        wandb.run.summary['training_time_minutes'] = round(training_time, 2)
        wandb.run.summary['best_epoch'] = best_epoch
        wandb.run.summary['best_val_loss'] = best_val_loss
        wandb.run.summary['estimated_mark'] = round(100 - 800 * best_val_loss, 1)
        wandb.finish()

    # Save training plot
    _save_training_plot(all_val_loss, args.model, exp_dir)

    print(f"\nNext step: python src/rocky/predict.py --exp_dir {exp_dir} --model {args.model}")


def _save_training_plot(val_losses, model_name, exp_dir):
    """Save a plot of validation loss over epochs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'b-o', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Validation MSE')
        plt.title(f'{model_name} — Validation Loss')
        plt.grid(True, alpha=0.3)

        # Mark the best epoch
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
        plt.annotate(f'Best: {best_loss:.5f} (epoch {best_epoch})',
                     xy=(best_epoch, best_loss),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, color='red')

        plt.tight_layout()
        plot_path = os.path.join(exp_dir, 'training_loss.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Training plot saved: {plot_path}")
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rocky\'s training script')

    # Model
    parser.add_argument('--model', type=str, default='nvidia',
                        choices=['nvidia', 'mobilenet'],
                        help='Model architecture (default: nvidia)')

    # Training — single phase (NVIDIA)
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max epochs for single-phase training (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')

    # Training — two-phase (MobileNet)
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs with frozen backbone (default: 5)')
    parser.add_argument('--finetune_epochs', type=int, default=15,
                        help='Fine-tuning epochs (default: 15)')
    parser.add_argument('--finetune_lr', type=float, default=1e-4,
                        help='Fine-tuning learning rate (default: 0.0001)')
    parser.add_argument('--unfreeze_layers', type=int, default=20,
                        help='Number of backbone layers to unfreeze (default: 20)')

    # Data
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split (default: 0.2)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory (default: data)')

    # Experiment
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (auto-generated if not set)')
    parser.add_argument('--description', type=str, default=None,
                        help='Experiment description for W&B')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')

    args = parser.parse_args()
    train(args)
