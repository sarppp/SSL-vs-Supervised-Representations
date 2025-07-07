import os
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from ..utils.logger_manager import DataSplitterLogger
from ..config import config_paths

def split_clean_dataset(pickle_path=config_paths.CLEAN_DATASET_PICKLE, test_size=0.15, val_size=0.15, random_state=42, base_data_dir=config_paths.BASE_DATA_DIR, 
                       few_shot_mode=None, few_shot_value=0.1, data_logger=None):
    """Load clean dataset and split into train/val/test with stratification.
    
    Args:
        pickle_path: Path to the clean dataset pickle file
        test_size: Proportion for test set (default 0.15 = 15%)
        val_size: Proportion for validation set (default 0.15 = 15%)
        random_state: Random seed for reproducibility
        base_data_dir: Base directory for relative paths (auto-detected if None)
        few_shot_mode: None, 'percentage', or 'per_class' for label hiding few-shot
        few_shot_value: Value for few-shot (0.01=1%, 0.1=10% for percentage; or samples per class)
        data_logger: DataSplitterLogger instance (will be created if None)
    
    Returns:
        tuple: (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    
    # Create logger if not provided
    if data_logger is None:
        data_logger = DataSplitterLogger("dataset")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both old format (tuple) and new format (dict)
    if isinstance(data, tuple):
        # Old format: (paths, labels)
        image_paths, labels = data
        use_relative_paths = False
        data_logger.log_dataset_loading(pickle_path, "legacy", len(image_paths))
        
        # ✅ ADDED: Convert old absolute paths to new paths if base_data_dir provided
        if base_data_dir is not None:
            data_logger.log_path_conversion(base_data_dir, "old_to_new")
            base_dir_name = os.path.basename(os.path.normpath(base_data_dir))
            new_paths = []
            for old_path in image_paths:
                try:
                    # More robust path parsing using pathlib for cross-platform compatibility
                    old_path_obj = Path(old_path)
                    
                    # Find the base directory in the path hierarchy
                    if base_dir_name in old_path_obj.parts:
                        # Get index of base directory in path parts
                        base_idx = old_path_obj.parts.index(base_dir_name)
                        # Extract relative path after base directory
                        rel_parts = old_path_obj.parts[base_idx + 1:]
                        if rel_parts:  # Only if there are parts after base directory
                            rel_path = os.path.join(*rel_parts)
                            new_path = os.path.join(base_data_dir, rel_path)
                            new_paths.append(new_path)
                        else:
                            new_paths.append(old_path)  # No relative path found
                    else:
                        # Base directory not found in path - check if it's already relative
                        if os.path.isabs(old_path):
                            data_logger.log_path_conversion_warning(old_path, base_dir_name)
                        new_paths.append(old_path)  # Keep original path
                except Exception as e:
                    # Fallback for any path parsing errors
                    data_logger.log_path_conversion_error(old_path, str(e))
                    new_paths.append(old_path)
            image_paths = new_paths
            
    else:
        # New format: dictionary with metadata
        image_paths = data['paths']
        labels = data['labels']
        use_relative_paths = data.get('use_relative_paths', False)
        stored_data_dir = data.get('data_dir', None)
        
        data_logger.log_dataset_loading(
            pickle_path, "metadata", data['total_images'], 
            data['num_classes'], data.get('corrupted_count', 0)
        )
        
        # Handle relative paths - convert back to absolute if needed
        if use_relative_paths:
            if base_data_dir is None:
                base_data_dir = stored_data_dir
            if base_data_dir is None:
                raise ValueError("Base data directory required for relative paths. Please provide base_data_dir parameter.")
            
            data_logger.log_path_conversion(base_data_dir, "relative")
            image_paths = [os.path.join(base_data_dir, rel_path) for rel_path in image_paths]
    
    # SORT to ensure consistent order across different systems
    sorted_data = sorted(zip(image_paths, labels))
    image_paths, labels = zip(*sorted_data)
    image_paths, labels = list(image_paths), list(labels)
    
    # Validate split proportions
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size ({test_size}) + val_size ({val_size}) must be < 1.0")
    
    train_size = 1.0 - test_size - val_size
    data_logger.log_split_configuration(train_size, val_size, test_size, len(image_paths))
    
    # Check class distribution before splitting
    original_distribution = Counter(labels)
    min_class_count = min(original_distribution.values())
    num_classes = len(original_distribution)
    
    # Ensure minimum samples per class for stratification
    min_samples_needed = 3  # At least 1 for each split
    if min_class_count < min_samples_needed:
        data_logger.log_stratification_warning(min_class_count)
    
    # Calculate minimum samples needed for each split to ensure no empty sets
    min_test_samples = max(1, num_classes)  # At least 1 sample per class for test
    min_val_samples = max(1, num_classes)   # At least 1 sample per class for val
    min_train_samples = max(1, num_classes) # At least 1 sample per class for train
    
    total_samples = len(image_paths)
    min_total_needed = min_test_samples + min_val_samples + min_train_samples
    
    # Check if we have enough samples
    if total_samples < min_total_needed:
        raise ValueError(f"Not enough samples ({total_samples}) for proper stratified split. "
                        f"Need at least {min_total_needed} samples for {num_classes} classes.")
    
    # Adjust split sizes if they would result in empty sets
    actual_test_size = max(int(total_samples * test_size), min_test_samples)
    actual_val_size = max(int(total_samples * val_size), min_val_samples)
    actual_train_size = total_samples - actual_test_size - actual_val_size
    
    if actual_train_size < min_train_samples:
        # Readjust if train becomes too small
        actual_train_size = min_train_samples
        remaining = total_samples - actual_train_size
        actual_test_size = max(remaining // 2, min_test_samples)
        actual_val_size = remaining - actual_test_size
        
        if actual_val_size < min_val_samples:
            actual_val_size = min_val_samples
            actual_test_size = remaining - actual_val_size
    
    # Convert back to proportions for sklearn
    adjusted_test_size = actual_test_size / total_samples
    adjusted_val_size = actual_val_size / total_samples
    
    print(f"📊 Adjusted split sizes:")
    print(f"   Original: test={test_size:.2%}, val={val_size:.2%}")
    print(f"   Adjusted: test={adjusted_test_size:.2%} ({actual_test_size} samples), val={adjusted_val_size:.2%} ({actual_val_size} samples)")
    print(f"   Train will be: {1-adjusted_test_size-adjusted_val_size:.2%} ({actual_train_size} samples)")
    
    # STEP 1: First split - separate test set
    try:
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, labels,
            test_size=adjusted_test_size,
            stratify=labels,
            random_state=random_state
        )
    except ValueError as e:
        print(f"⚠️  Stratification failed: {e}")
        print("   Falling back to non-stratified split...")
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, labels,
            test_size=adjusted_test_size,
            random_state=random_state
        )

    # STEP 2: Second split - separate train and validation
    val_size_from_remaining = adjusted_val_size / (1 - adjusted_test_size)
    try:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_size_from_remaining,
            stratify=train_val_labels,
            random_state=random_state
        )
    except ValueError as e:
        print(f"⚠️  Stratification failed for train/val split: {e}")
        print("   Falling back to non-stratified split...")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_size_from_remaining,
            random_state=random_state
        )
    
    # Final validation - ensure no empty sets
    if len(test_paths) == 0:
        raise ValueError("❌ Test set is empty! This should not happen after adjustments.")
    if len(val_paths) == 0:
        raise ValueError("❌ Validation set is empty! This should not happen after adjustments.")
    if len(train_paths) == 0:
        raise ValueError("❌ Training set is empty! This should not happen after adjustments.")

    # Print detailed split statistics
    total_images = len(image_paths)
    data_logger.log_split_summary(len(train_paths), len(val_paths), len(test_paths), total_images)

    # Verify class distribution across splits
    all_classes = set(labels)
    train_classes = set(train_labels)
    val_classes = set(val_labels)
    test_classes = set(test_labels)
    
    missing_train = all_classes - train_classes
    missing_val = all_classes - val_classes
    missing_test = all_classes - test_classes

    data_logger.log_class_distribution_verification(
        len(all_classes), len(train_classes), len(val_classes), len(test_classes),
        missing_train, missing_val, missing_test
    )

    # Show per-class distribution for verification
    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)
    test_dist = Counter(test_labels)
    
    data_logger.log_per_class_distribution(list(all_classes), train_dist, val_dist, test_dist)
    # Quick data inspection
    data_logger.log_data_inspection(train_paths, train_labels, val_paths, test_paths)
    
    # Apply label hiding few-shot if specified
    if few_shot_mode is not None:
        train_paths, train_labels = apply_label_hiding_few_shot(train_paths, train_labels, few_shot_mode, few_shot_value, random_state, data_logger)
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def apply_label_hiding_few_shot(train_paths, train_labels, mode=None, value=0.1, random_state=42, data_logger=None):
    """
    True few-shot learning: Keep ALL training images but hide most labels.
    
    Args:
        train_paths: All training image paths
        train_labels: All training labels  
        mode: 'percentage' or 'per_class'
        value: Percentage of labeled data or samples per class
        random_state: Random seed
    
    Returns:
        train_paths: ALL paths (unchanged)
        masked_labels: Labels with most set to -1 (unlabeled)
    """
    if mode is None:
        if data_logger:
            data_logger.log_few_shot_disabled(len(train_paths))
        return train_paths, train_labels
    
    if data_logger:
        data_logger.log_few_shot_enabled(mode, value)
    
    total_samples = len(train_paths)
    
    if mode == 'percentage':
        # Label only X% of data
        n_labeled = int(total_samples * value)
        labeled_indices = np.random.RandomState(random_state).choice(
            total_samples, n_labeled, replace=False
        )
        
    elif mode == 'per_class':
        # Label only X samples per class
        labeled_indices = []
        for class_name in set(train_labels):
            class_indices = [i for i, label in enumerate(train_labels) if label == class_name]
            n_take = min(int(value), len(class_indices))
            selected = np.random.RandomState(random_state).choice(
                class_indices, n_take, replace=False
            )
            labeled_indices.extend(selected)
        labeled_indices = np.array(labeled_indices)
    
    # Create masked labels: -1 for unlabeled, original for labeled
    masked_labels = [-1] * total_samples  # All unlabeled initially
    for idx in labeled_indices:
        masked_labels[idx] = train_labels[idx]  # Restore original label
    
    # Log detailed results for verification
    if data_logger:
        labeled_class_counts = Counter([train_labels[i] for i in labeled_indices])
        data_logger.log_few_shot_results(total_samples, len(labeled_indices), labeled_class_counts)
    
    # Additional detailed logging for verification
    total_images = len(train_paths)
    total_labels = len(train_labels)
    unique_classes = len(set(train_labels))
    labeled_samples = len(labeled_indices)
    unlabeled_samples = total_samples - labeled_samples
    
    print(f"🎯 LABEL HIDING FEW-SHOT VERIFICATION:")
    print(f"   📊 Total images: {total_images:,}")
    print(f"   📊 Total labels: {total_labels:,}")
    print(f"   📊 Unique classes: {unique_classes}")
    print(f"   ✅ Labeled samples: {labeled_samples:,} ({labeled_samples/total_samples*100:.1f}%)")
    print(f"   ❌ Unlabeled samples: {unlabeled_samples:,} ({unlabeled_samples/total_samples*100:.1f}%)")
    print(f"   🔍 Sanity check: images={total_images} == labels={total_labels} == samples={total_samples}")
    
    # Show per-class labeled distribution
    labeled_class_counts = Counter([train_labels[i] for i in labeled_indices])
    print(f"   📈 Labeled samples per class:")
    for class_name, count in sorted(labeled_class_counts.items()):
        print(f"      {class_name}: {count} samples")
    
    return train_paths, masked_labels

def split_clean_dataset_with_config(pickle_path, config_module, base_data_dir=config_paths.BASE_DATA_DIR):
    """Load and split dataset using config module settings (including label hiding few-shot)."""
    few_shot_mode = getattr(config_module, 'FEW_SHOT_MODE', None)
    few_shot_value = getattr(config_module, 'FEW_SHOT_VALUE', 0.1)
    
    return split_clean_dataset(
        pickle_path=pickle_path,
        test_size=config_module.TEST_SIZE,
        val_size=config_module.VAL_SIZE,
        random_state=config_module.RANDOM_STATE,
        base_data_dir=base_data_dir,
        few_shot_mode=few_shot_mode,
        few_shot_value=few_shot_value
    )