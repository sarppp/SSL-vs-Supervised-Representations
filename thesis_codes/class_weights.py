import numpy as np
import torch
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(train_labels, class_names):
    """Calculate class weights for handling imbalanced datasets."""
    print(f"⚖️  Calculating class weights for balanced training...")
    
    # Convert all labels to strings to handle mixed int/string labels
    string_train_labels = [str(label) for label in train_labels]
    
    # Calculate class weights
    unique_labels = sorted(list(set(string_train_labels)))
    unique_labels_array = np.array(unique_labels)
    train_labels_array = np.array(string_train_labels)

    class_weights = compute_class_weight('balanced', classes=unique_labels_array, y=train_labels_array)
    class_weights_dict = dict(zip(unique_labels, class_weights))

    # Convert to tensor - order must match class_to_idx
    ordered_weights = [class_weights_dict[class_name] for class_name in class_names]
    class_weights_tensor = torch.FloatTensor(ordered_weights)

    print(f"✅ Class weights calculated:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {ordered_weights[i]:.3f}")
    
    print(f"📊 Class weights tensor shape: {class_weights_tensor.shape}")
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
        print("✅ All datasets have consistent class structure!")
        return True
    else:
        print("⚠️  Warning: Inconsistent class structure across datasets!")
        return False