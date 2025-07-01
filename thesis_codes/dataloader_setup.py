from torch.utils.data import DataLoader
import custom_dataset
import transform_utils

def create_dataloaders(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, config_module=None, run_batch_test=False, test_keep_original_size=False):
    """Create DataLoaders for train/val/test splits with config-based settings."""
    
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        import config
        config_module = config
    
    # Get transforms
    train_transform = transform_utils.get_train_transforms(config_module)
    val_transform = transform_utils.get_val_transforms(config_module)
    test_transform = transform_utils.get_test_transforms(config_module, keep_original_size=test_keep_original_size)
    
    # Create datasets
    train_dataset = custom_dataset.CustomCropDataset(train_paths, train_labels, transform=train_transform, fallback_size=config_module.IMAGE_SIZE)
    val_dataset = custom_dataset.CustomCropDataset(val_paths, val_labels, transform=val_transform, fallback_size=config_module.IMAGE_SIZE)
    test_dataset = custom_dataset.CustomCropDataset(test_paths, test_labels, transform=test_transform, fallback_size=config_module.IMAGE_SIZE)
    
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
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=config_module.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=config_module.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=config_module.NUM_WORKERS)
    
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