import os
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from logger_manager import DataSplitterLogger

def split_clean_dataset(pickle_path, test_size=0.15, val_size=0.15, random_state=42, base_data_dir=None, 
                       few_shot_mode=None, few_shot_value=0.1, data_logger=None):
    """Load clean dataset and split into train/val/test with stratification.
    
    Args:
        pickle_path: Path to the clean dataset pickle file
        test_size: Proportion for test set (default 0.15 = 15%)
        val_size: Proportion for validation set (default 0.15 = 15%)
        random_state: Random seed for reproducibility
        base_data_dir: Base directory for relative paths (auto-detected if None)
        few_shot_mode: None, 'percentage', or 'per_class' for few-shot learning
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
    
    # Ensure minimum samples per class for stratification
    min_samples_needed = 3  # At least 1 for each split
    if min_class_count < min_samples_needed:
        data_logger.log_stratification_warning(min_class_count)
    
    # STEP 1: First split - separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # STEP 2: Second split - separate train and validation
    val_size_from_remaining = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size_from_remaining,
        stratify=train_val_labels,
        random_state=random_state
    )

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
    
    # Apply few-shot learning if specified
    if few_shot_mode is not None:
        train_paths, train_labels = apply_few_shot(train_paths, train_labels, few_shot_mode, few_shot_value, random_state, data_logger)
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def split_clean_dataset_with_config(pickle_path, config_module, base_data_dir=None):
    """Load and split dataset using config module settings (including few-shot)."""
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

def apply_few_shot(train_paths, train_labels, mode=None, value=0.1, random_state=42, data_logger=None):
    """Simulate few-shot learning by using only a subset of labeled training data.
    
    This simulates real-world scenarios where you have limited labeled data,
    comparing how CNN vs DINO perform with scarce labels.
    """
    original_count = len(train_paths)
    
    if mode is None:
        if data_logger:
            data_logger.log_few_shot_disabled(original_count)
        return train_paths, train_labels
    
    if data_logger:
        data_logger.log_few_shot_enabled(mode, value)
    
    if mode == 'percentage':
        # Use percentage of total training data
        n_samples = int(len(train_paths) * value)
        indices = np.random.RandomState(random_state).choice(len(train_paths), n_samples, replace=False)
    elif mode == 'per_class':
        # Use fixed number of samples per class
        indices = []
        for class_name in set(train_labels):
            class_indices = [i for i, label in enumerate(train_labels) if label == class_name]
            n_take = min(int(value), len(class_indices))
            selected = np.random.RandomState(random_state).choice(class_indices, n_take, replace=False)
            indices.extend(selected)
        indices = np.array(indices)
    
    # Apply selection
    few_shot_paths = [train_paths[i] for i in indices]
    few_shot_labels = [train_labels[i] for i in indices]
    
    # Log results
    if data_logger:
        class_counts = Counter(few_shot_labels)
        data_logger.log_few_shot_results(original_count, len(few_shot_paths), class_counts)
    
    return few_shot_paths, few_shot_labels