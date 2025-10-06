import numpy as np
import torch
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(train_labels, class_names):
    """Calculate class weights for handling imbalanced datasets."""
    print(f"Calculating class weights for balanced training...")
    
    # Convert all labels to strings to handle mixed int/string labels
    string_train_labels = [str(label) for label in train_labels]
    
    # Check for zero-shot case (all labels are -1)
    valid_labels = [label for label in string_train_labels if label != '-1']
    
    if not valid_labels:
        print(f"🚨 ZERO-SHOT MODE: No valid labels found, using uniform weights")
        # Return uniform weights for all classes
        uniform_weights = [1.0] * len(class_names)
        class_weights_tensor = torch.FloatTensor(uniform_weights)
        print(f"Uniform class weights applied: {uniform_weights[0]:.3f} for all {len(class_names)} classes")
        return class_weights_tensor
    
    # Calculate class weights for valid labels only
    unique_labels = sorted(list(set(valid_labels)))
    unique_labels_array = np.array(unique_labels)
    valid_labels_array = np.array(valid_labels)

    class_weights = compute_class_weight('balanced', classes=unique_labels_array, y=valid_labels_array)
    class_weights_dict = dict(zip(unique_labels, class_weights))

    # Convert to tensor - order must match class_to_idx
    ordered_weights = []
    for class_name in class_names:
        if str(class_name) in class_weights_dict:
            ordered_weights.append(class_weights_dict[str(class_name)])
        else:
            # If class not in labeled data, use average weight
            avg_weight = np.mean(class_weights) if len(class_weights) > 0 else 1.0
            ordered_weights.append(avg_weight)
    
    class_weights_tensor = torch.FloatTensor(ordered_weights)

    print(f"Class weights calculated:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {ordered_weights[i]:.3f}")
    
    print(f"Class weights tensor shape: {class_weights_tensor.shape}")
    return class_weights_tensor

def analyze_class_distribution(train_labels):
    """Analyze class distribution and imbalance in training data."""
    print(f"\n=== CLASS DISTRIBUTION ANALYSIS ===")
    
    train_class_counts = Counter(train_labels)
    print("Training set class distribution:")
    
    total_samples = len(train_labels)
    for class_name, count in sorted(train_class_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    return train_class_counts

def verify_dataset_consistency(train_dataset, val_dataset, test_dataset):
    """Verify all datasets have consistent class structure."""
    print(f"\n=== CLASS CONSISTENCY CHECK ===")
    print(f"Train dataset classes: {len(train_dataset.classes)}")
    print(f"Val dataset classes: {len(val_dataset.classes)}")
    print(f"Test dataset classes: {len(test_dataset.classes)}")

    if (train_dataset.classes == val_dataset.classes == test_dataset.classes and
        train_dataset.class_to_idx == val_dataset.class_to_idx == test_dataset.class_to_idx):
        print("All datasets have consistent class structure!")
        return True
    else:
        print("WARNING: Inconsistent class structure across datasets!")
        return False