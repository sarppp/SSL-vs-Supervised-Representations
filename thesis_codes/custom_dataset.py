import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomCropDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, fallback_size=(224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.fallback_size = fallback_size

        # Create a mapping from class names to integer labels
        self.classes = sorted(list(set(labels)))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.targets = [self.class_to_idx[label] for label in labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.targets[idx]

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', self.fallback_size, (0, 0, 0))
            print(f"Using black image fallback for {img_path}")

        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                print(f"Error applying transform to {img_path}: {e}")
                # If transform fails, create a black tensor
                channels = 3
                height, width = self.fallback_size
                img = torch.zeros(channels, height, width)

        return img, label
