#!/usr/bin/env python3
"""
True Few-Shot Learning: Hide labels instead of reducing dataset size
"""
import numpy as np
from collections import Counter

def apply_label_hiding_few_shot(train_paths, train_labels, mode='percentage', value=0.1, random_state=42):
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
        labeled_mask: Boolean mask showing which samples have labels
    """
    
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
    
    # Create boolean mask for easy filtering
    labeled_mask = [i in labeled_indices for i in range(total_samples)]
    
    print(f"🎯 Label Hiding Few-Shot Applied:")
    print(f"   Total samples: {total_samples}")
    print(f"   Labeled samples: {len(labeled_indices)} ({len(labeled_indices)/total_samples*100:.1f}%)")
    print(f"   Unlabeled samples: {total_samples - len(labeled_indices)}")
    
    # Show per-class labeled distribution
    labeled_class_counts = Counter([train_labels[i] for i in labeled_indices])
    print(f"   Labeled per class: {dict(labeled_class_counts)}")
    
    return train_paths, masked_labels, labeled_mask

def create_few_shot_loss_function(criterion, ignore_index=-1):
    """
    Create loss function that ignores unlabeled samples (-1)
    """
    def few_shot_loss(outputs, targets):
        # Find samples with valid labels (not -1)
        valid_mask = targets != ignore_index
        
        if valid_mask.sum() == 0:
            # No labeled samples in this batch
            return torch.tensor(0.0, requires_grad=True)
        
        # Compute loss only on labeled samples
        valid_outputs = outputs[valid_mask]
        valid_targets = targets[valid_mask]
        
        return criterion(valid_outputs, valid_targets)
    
    return few_shot_loss

# Example usage in training loop:
def train_with_hidden_labels(model, train_loader, optimizer, device):
    """
    Training loop that handles hidden labels (-1)
    """
    import torch
    
    model.train()
    total_loss = 0
    labeled_samples = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    few_shot_criterion = create_few_shot_loss_function(criterion)
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Count labeled samples in this batch
        labeled_in_batch = (targets != -1).sum().item()
        labeled_samples += labeled_in_batch
        
        if labeled_in_batch == 0:
            # Skip batch if no labeled samples
            continue
            
        optimizer.zero_grad()
        outputs = model(data)
        
        # Use few-shot loss function
        loss = few_shot_criterion(outputs, targets)
        
        if loss.item() > 0:  # Only backprop if there was a valid loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss, labeled_samples

if __name__ == "__main__":
    # Example usage
    import torch
    
    # Mock data
    train_paths = [f"image_{i}.jpg" for i in range(1000)]
    train_labels = [f"class_{i%10}" for i in range(1000)]  # 10 classes
    
    # Apply label hiding (10% labeled)
    paths, masked_labels, labeled_mask = apply_label_hiding_few_shot(
        train_paths, train_labels, mode='percentage', value=0.1
    )
    
    print(f"\nFirst 20 samples:")
    for i in range(20):
        status = "LABELED" if labeled_mask[i] else "UNLABELED"
        print(f"  {paths[i]}: {masked_labels[i]} ({status})") 