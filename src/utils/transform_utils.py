import random
from torchvision import transforms
from torchvision.transforms import functional as F

def get_train_transforms(config_module=None):
    """Get enhanced training transforms from config"""
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        from ..config import config
        config_module = config
    
    print("Creating enhanced training transforms:")
    
    # Explicitly type as list[object] to silence type checker errors
    transform_list: list = [
        transforms.Resize(config_module.TRAIN_TRANSFORMS['resize'])
    ]
    print(f"  Resize: {config_module.TRAIN_TRANSFORMS['resize']}")
    
    # Geometric augmentations
    if config_module.TRAIN_TRANSFORMS.get('rotation'):
        rotation_deg = config_module.TRAIN_TRANSFORMS['rotation']
        transform_list.append(transforms.RandomRotation(rotation_deg))
        print(f"  Random Rotation: ±{rotation_deg}°")
    
    if config_module.TRAIN_TRANSFORMS.get('affine'):
        affine = config_module.TRAIN_TRANSFORMS['affine']
        transform_list.append(transforms.RandomAffine(
            degrees=0, 
            translate=affine['translate'], 
            scale=affine['scale']
        ))
        print(f"  Random Affine: translate={affine['translate']}, scale={affine['scale']}")
    
    if config_module.TRAIN_TRANSFORMS.get('horizontal_flip'):
        transform_list.append(transforms.RandomHorizontalFlip())
        print(f"  Random Horizontal Flip: enabled")
    
    if config_module.TRAIN_TRANSFORMS.get('vertical_flip'):
        vflip_prob = config_module.TRAIN_TRANSFORMS['vertical_flip']
        transform_list.append(transforms.RandomVerticalFlip(p=vflip_prob))
        print(f"  Random Vertical Flip: {vflip_prob*100}% chance")
    
    # Color augmentations
    if config_module.TRAIN_TRANSFORMS.get('color_jitter'):
        cj = config_module.TRAIN_TRANSFORMS['color_jitter']
        transform_list.append(transforms.ColorJitter(
            brightness=cj['brightness'],
            contrast=cj['contrast'], 
            saturation=cj['saturation'],
            hue=cj['hue']
        ))
        print(f"  Color Jitter: brightness={cj['brightness']}, contrast={cj['contrast']}, saturation={cj['saturation']}, hue={cj['hue']}")
    
    # Extra brightness/contrast randomization
    if config_module.TRAIN_TRANSFORMS.get('extra_brightness'):
        extra_bright = config_module.TRAIN_TRANSFORMS['extra_brightness']
        transform_list.append(
            transforms.Lambda(
                lambda x: F.adjust_brightness(x, 1 + random.uniform(-extra_bright, extra_bright))
            )  # type: ignore
        )
        print(f"  Extra Brightness Variation: ±{extra_bright*100}%")
    
    if config_module.TRAIN_TRANSFORMS.get('extra_contrast'):
        extra_contrast = config_module.TRAIN_TRANSFORMS['extra_contrast']
        transform_list.append(
            transforms.Lambda(
                lambda x: F.adjust_contrast(x, 1 + random.uniform(-extra_contrast, extra_contrast))
            )  # type: ignore
        )
        print(f"  Extra Contrast Variation: ±{extra_contrast*100}%")
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    print(f"  ToTensor: convert to tensor")
    
    # Normalization
    if config_module.TRAIN_TRANSFORMS.get('normalize'):
        norm = config_module.TRAIN_TRANSFORMS['normalize']
        transform_list.append(transforms.Normalize(mean=norm['mean'], std=norm['std']))
        print(f"  Normalize: mean={norm['mean']}, std={norm['std']}")
    
    print(f"Total training transforms: {len(transform_list)}")
    print("Enhanced for crop pest detection with aggressive augmentation!")
    
    return transforms.Compose(transform_list)

def get_val_transforms(config_module=None):
    """Get validation transforms from config - ALWAYS resize, no augmentation"""
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        from ..config import config
        config_module = config
    
    print("\nCreating validation transforms:")
    
    transform_list: list = [
        transforms.Resize(config_module.VAL_TEST_TRANSFORMS['resize']),
        transforms.ToTensor()
    ]
    print(f"  Resize: {config_module.VAL_TEST_TRANSFORMS['resize']}")
    print(f"  ToTensor: convert to tensor")
    
    # Normalization
    if config_module.VAL_TEST_TRANSFORMS.get('normalize'):
        norm = config_module.VAL_TEST_TRANSFORMS['normalize']
        transform_list.append(transforms.Normalize(mean=norm['mean'], std=norm['std']))
        print(f"  Normalize: mean={norm['mean']}, std={norm['std']}")
    
    print(f"Total validation transforms: {len(transform_list)}")
    
    return transforms.Compose(transform_list)

def get_test_transforms(config_module=None, keep_original_size=False):
    """Get test transforms with option to keep original size"""
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        from ..config import config
        config_module = config
    
    print("\nCreating test transforms:")
    
    if keep_original_size:
        transform_list: list = [transforms.ToTensor()]
        print(f"  Keep original size: enabled")
        print(f"  ToTensor: convert to tensor")
        
        # Still normalize even at original size
        if config_module.VAL_TEST_TRANSFORMS.get('normalize'):
            norm = config_module.VAL_TEST_TRANSFORMS['normalize']
            transform_list.append(transforms.Normalize(mean=norm['mean'], std=norm['std']))
            print(f"  Normalize: mean={norm['mean']}, std={norm['std']}")
        
        print(f"Total test transforms: {len(transform_list)} (original size)")
        return transforms.Compose(transform_list)
    else:
        print("  Using validation transforms for test set (resize enabled)")
        transforms_obj = get_val_transforms(config_module)  # This will print its own details
        return transforms_obj

# For backward compatibility
def get_val_test_transforms():
    """Deprecated: Use get_val_transforms() instead"""
    return get_val_transforms()