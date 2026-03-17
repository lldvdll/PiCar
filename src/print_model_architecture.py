import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large

def save_model_map(model_class, model_name, filename):
    print(f"Generating map for {model_name}...")
    # Load the base model just like you do in your training script
    base_model = model_class(input_shape=(96, 160, 3), include_top=False, weights='imagenet')
    
    filename = os.path.join("ref", filename)
    with open(filename, 'w') as f:
        f.write(f"INDEX | LAYER NAME ({model_name})\n")
        f.write("-" * 50 + "\n")
        
        for i, layer in enumerate(base_model.layers):
            # MobileNetV2 specific logic for safe cuts
            if model_name == "MobileNetV2":
                if layer.name.endswith('_add') or (layer.name.endswith('_project_BN') and 'add' not in base_model.layers[i+1].name if i+1 < len(base_model.layers) else False):
                    f.write(f"{i:5d} | ---> [SAFE CUT POINT] {layer.name}\n")
                    f.write("-" * 50 + "\n")
                elif 'expand_relu' in layer.name:
                    f.write(f"{i:5d} | {layer.name}  (Start of new block)\n")
                else:
                    f.write(f"{i:5d} | {layer.name}\n")
            
            # For V1 and V3, print standard layer names
            else:
                f.write(f"{i:5d} | {layer.name}\n")
                
    print(f"  -> Saved to {filename}\n")

if __name__ == "__main__":
    save_model_map(MobileNet, "MobileNet", "MobileNet_architecture.txt")
    save_model_map(MobileNetV2, "MobileNetV2", "MobileNetV2_architecture.txt")
    save_model_map(MobileNetV3Small, "MobileNetV3Small", "MobileNetV3Small_architecture.txt")
    save_model_map(MobileNetV3Large, "MobileNetV3Large", "MobileNetV3Large_architecture.txt")