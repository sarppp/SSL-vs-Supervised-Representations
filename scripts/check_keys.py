def inspect_checkpoint(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu')
    for k, v in checkpoint.items():
        if isinstance(v, (int, float, str, tuple, list)):
            print(f"{k}: {v}")
        else:
            print(f"{k}: <{type(v)}>")

inspect_checkpoint('models/best_dino_vits14_acc71.01_20250707_095442.pth')