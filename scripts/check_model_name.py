import torch
import sys

def print_checkpoint_info(path):
    ckpt = torch.load(path, map_location='cpu')
    print(f"File: {path}")
    print(f"  model_name: {ckpt.get('model_name', 'N/A')}")
    print(f"  val_acc: {ckpt.get('val_acc', 'N/A')}")
    print(f"  class_names: {ckpt.get('class_names', 'N/A')}")
    print()

if __name__ == "__main__":
    import glob
    import os

    # Default: check all .pth files in models/
    paths = glob.glob("models/*.pth")
    if len(sys.argv) > 1:
        paths = sys.argv[1:]

    for path in paths:
        if os.path.isfile(path):
            try:
                print_checkpoint_info(path)
            except Exception as e:
                print(f"  [ERROR] Could not read {path}: {e}")
