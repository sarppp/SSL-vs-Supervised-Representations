from torch.utils.data import DataLoader
from . import custom_dataset
from ..utils import transform_utils
from ..config import config_paths
import torch
from collections import Counter

def create_dataloaders(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, config_module=None, run_batch_test=False, test_keep_original_size=False):
    """Create DataLoaders for train/val/test splits with config-based settings."""
    
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        from ..config import config
        config_module = config
    
    # Get transforms
    train_transform = transform_utils.get_train_transforms(config_module)
    val_transform = transform_utils.get_val_transforms(config_module)
    test_transform = transform_utils.get_test_transforms(config_module, keep_original_size=test_keep_original_size)
    
    # ------------------------------------------------------------------
    # Build a *shared* label→index mapping so that **all** dataset splits
    # use identical indices.  Crucially, the special ``ignore_index`` (-1)
    # that denotes *unlabelled* samples for few-shot experiments is *excluded*
    # from the mapping so it is never counted as its own class.
    # ------------------------------------------------------------------
    IGNORE_INDEX = -1
    all_labels_union = set([lbl for lbl in (train_labels + val_labels + test_labels) if lbl != IGNORE_INDEX])
    class_to_idx = {lbl: idx for idx, lbl in enumerate(sorted(all_labels_union))}

    # Create datasets – pass the shared mapping so every split is consistent
    train_dataset = custom_dataset.CustomCropDataset(
        train_paths,
        train_labels,
        transform=train_transform,
        fallback_size=config_module.IMAGE_SIZE,
        class_to_idx=class_to_idx,
        ignore_index=IGNORE_INDEX,
        validate_images=getattr(config_module, 'VALIDATE_IMAGES', True),
    )

    val_dataset = custom_dataset.CustomCropDataset(
        val_paths,
        val_labels,
        transform=val_transform,
        fallback_size=config_module.IMAGE_SIZE,
        class_to_idx=class_to_idx,
        ignore_index=IGNORE_INDEX,
        validate_images=getattr(config_module, 'VALIDATE_IMAGES', True),
    )

    test_dataset = custom_dataset.CustomCropDataset(
        test_paths,
        test_labels,
        transform=test_transform,
        fallback_size=config_module.IMAGE_SIZE,
        class_to_idx=class_to_idx,
        ignore_index=IGNORE_INDEX,
        validate_images=getattr(config_module, 'VALIDATE_IMAGES', True),
    )
    
    # 🔧 FIX: Use smaller batch size for large test sets to reduce memory pressure
    train_batch_size = config_module.BATCH_SIZE
    val_batch_size = config_module.BATCH_SIZE
    test_batch_size = config_module.BATCH_SIZE
    
    # Reduce test batch size if test set is large to prevent memory issues
    if len(test_dataset) > 1000:
        test_batch_size = min(16, config_module.BATCH_SIZE)  # Max 16 for large test sets
        print(f"🔧 Large test set detected ({len(test_dataset)} samples)")
        print(f"   Reducing test batch size: {config_module.BATCH_SIZE} → {test_batch_size}")
    
    # Create DataLoaders
    # Note on drop_last=True for train_loader:
    # This is crucial to prevent crashes when using layers like BatchNorm1d.
    # If the last batch of an epoch contains only a single sample, BatchNorm1d will fail
    # because it cannot compute batch statistics (mean/variance) on a single item.
    # This is especially common with small or few-shot datasets.
    # While this does discard a tiny fraction of data from each epoch (the last incomplete batch),
    # it is a standard and necessary practice for robust training. The impact on performance is negligible.
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=config_module.NUM_WORKERS, drop_last=True)
    
    # For validation and test sets, we don't drop the last batch as the model is in eval mode
    # and batch normalization layers are not being updated. This ensures we evaluate on the entire dataset.
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=config_module.NUM_WORKERS, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=config_module.NUM_WORKERS, drop_last=False)
    
    # Print summary
    total_images = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print(f"\n=== DATASET SUMMARY ===")
    print(f"Training: {len(train_dataset)} samples ({len(train_dataset)/total_images*100:.1f}%)")
    print(f"Validation: {len(val_dataset)} samples ({len(val_dataset)/total_images*100:.1f}%)")
    print(f"Test: {len(test_dataset)} samples ({len(test_dataset)/total_images*100:.1f}%)")
    print(f"Batch sizes: Train={train_batch_size}, Val={val_batch_size}, Test={test_batch_size}")
    print(f"Num workers: {config_module.NUM_WORKERS}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Optional batch test
    if run_batch_test:
        print(f"\n=== BATCH LOADING TEST ===")
        try:
            train_batch = next(iter(train_loader))
            train_images, train_labels_batch = train_batch
            print(f"✅ Batch test successful!")
            print(f"   Image shape: {train_images.shape}")
            print(f"   Label shape: {train_labels_batch.shape}")
            print(f"   Image range: [{train_images.min():.3f}, {train_images.max():.3f}]")
        except Exception as e:
            print(f"❌ Batch loading error: {e}")
    else:
        print(f"\n✅ DataLoaders ready for training!")
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def get_class_info(train_labels, device):
    """Calculate class weights and names from training labels, ignoring few-shot placeholders."""
    
    # CRITICAL FIX: Explicitly filter out the ignore_index (-1) before determining class info.
    # This ensures that the number of classes is based on the true labels, not the placeholders
    # used for few-shot learning.
    true_labels = [label for label in train_labels if label != -1]
    
    if not true_labels:
        # Handle edge case where a split might have no labeled data at all
        # This is unlikely but robust to handle. We assume original classes from all labels.
        class_names = sorted(list(set(train_labels) - {-1}))
        if not class_names: # If truly no labels other than -1
             raise ValueError("Cannot determine class info: no valid labels found.")
        num_classes = len(class_names)
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        class_weights = torch.ones(num_classes) # No information, so equal weights
    else:
        # Proceed with normal calculation on the filtered true labels
        class_names = sorted(list(set(true_labels)))
        num_classes = len(class_names)
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        # Calculate class weights from the distribution of the true labels
        class_counts = Counter(true_labels)
        total_samples = len(true_labels)
        
        class_weights = torch.zeros(num_classes)
        for i, class_name in enumerate(class_names):
            count = class_counts.get(class_name, 0)
            if count > 0:
                class_weights[i] = total_samples / (num_classes * count)
            else:
                # This case should not be hit if class_names is derived from true_labels
                class_weights[i] = 1.0

    return class_names, class_to_idx, class_weights.to(device)