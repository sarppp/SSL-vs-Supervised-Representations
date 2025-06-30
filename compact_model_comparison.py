#!/usr/bin/env python3
"""
🥊 Compact Model Comparison - CNN vs DINOv2
Super simple 2-model comparison in <100 lines
"""
import torch
import sys
import os
import time
from torch.utils.data import Subset
import numpy as np

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # Import modules
    import data_splitter # type: ignore
    import dataloader_setup # type: ignore
    import config # type: ignore
    import config_dinov2 # type: ignore
    import model_setup # type: ignore
    import training # type: ignore
    import evaluation # type: ignore
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")

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
            return torch.tensor(0.0, requires_grad=True, device=outputs.device)
        
        # Compute loss only on labeled samples
        valid_outputs = outputs[valid_mask]
        valid_targets = targets[valid_mask]
        
        return criterion(valid_outputs, valid_targets)
    
    return few_shot_loss

def run_model(model_type='cnn', sample_size=None, few_shot_mode=None, few_shot_value=0.1):
    """Run single model training and return results"""
    print(f"\n{'='*50}")
    print(f"🧪 Testing {model_type.upper()}")
    print(f"{'='*50}")
    
    # Select config (make a copy to avoid modifying original)
    import copy
    active_config = copy.deepcopy(config_dinov2 if model_type == 'dinov2' else config)
    model_name = active_config.MODEL_NAME
    
    # Quick config overrides for fast testing
    active_config.EPOCHS = 3
    active_config.BATCH_SIZE = min(16, active_config.BATCH_SIZE)
    
    print(f"🔧 Model: {model_name}")
    print(f"⚙️  Epochs: {active_config.EPOCHS}, Batch: {active_config.BATCH_SIZE}")
    
    start_time = time.time()
    
    try:
        # Load data (use existing clean dataset) - NO few-shot here, we'll apply label hiding later
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = data_splitter.split_clean_dataset(
            pickle_path='/teamspace/studios/this_studio/thesis_codes/clean_dataset.pkl',
            base_data_dir='/teamspace/studios/this_studio/crop_pest_data',
            few_shot_mode=None  # Don't reduce dataset size
        )
        
        # 🎯 TRUE FEW-SHOT: Hide labels instead of reducing dataset size
        original_train_size = len(train_paths)
        labeled_samples_count = original_train_size
        
        if few_shot_mode is not None:
            train_paths, train_labels, labeled_mask = apply_label_hiding_few_shot(
                train_paths, train_labels, few_shot_mode, few_shot_value, 
                active_config.RANDOM_STATE if hasattr(active_config, 'RANDOM_STATE') else 42
            )
            labeled_samples_count = sum(labeled_mask)
            print(f"🎯 Label Hiding Applied: {labeled_samples_count}/{original_train_size} samples have labels")
        else:
            print(f"🎯 Few-shot disabled: All {original_train_size} samples have labels")
        
        # Create dataloaders
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = dataloader_setup.create_dataloaders(
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
            config_module=active_config, run_batch_test=False
        )
        
        # Get original dataset info before subsetting
        original_train_dataset = train_dataset
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes
        class_to_idx = train_dataset.class_to_idx
        
        # 📊 SAMPLE SIZE LIMITING (if requested)
        if sample_size and sample_size < len(train_loader.dataset):
            train_size = min(sample_size * 70 // 100, len(train_loader.dataset))
            val_size = min(sample_size * 15 // 100, len(val_loader.dataset))
            test_size = min(sample_size * 15 // 100, len(test_loader.dataset))
            
            # Ensure minimum sizes
            train_size = max(train_size, num_classes)
            val_size = max(val_size, 1)
            test_size = max(test_size, 1)
            
            train_indices = np.random.choice(len(train_loader.dataset), size=train_size, replace=False).tolist()
            val_indices = np.random.choice(len(val_loader.dataset), size=val_size, replace=False).tolist()
            test_indices = np.random.choice(len(test_loader.dataset), size=test_size, replace=False).tolist()
            
            from torch.utils.data import DataLoader
            train_loader = DataLoader(Subset(train_dataset, train_indices), 
                                    batch_size=active_config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(Subset(val_dataset, val_indices), 
                                  batch_size=active_config.BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(Subset(test_dataset, test_indices), 
                                   batch_size=active_config.BATCH_SIZE, shuffle=False)
            
            print(f"📊 Dataset limited to: Train={train_size}, Val={val_size}, Test={test_size}")
            # Update labeled samples count if we reduced dataset size after few-shot
            if few_shot_mode is not None and sample_size < original_train_size:
                labeled_samples_count = min(labeled_samples_count, train_size)
        
        # Create model
        model = model_setup.create_model(num_classes, model_name, active_config).to(device)
        
        print(f"🎯 Classes: {num_classes}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=active_config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        
        # 🎯 CREATE FEW-SHOT AWARE LOSS FUNCTION
        base_criterion = torch.nn.CrossEntropyLoss()
        if few_shot_mode is not None:
            criterion = create_few_shot_loss_function(base_criterion, ignore_index=-1)
            print(f"🎯 Using few-shot loss function (ignores -1 labels)")
        else:
            criterion = base_criterion
            print(f"🎯 Using standard loss function")
        
        # Train model
        train_result = training.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_name=model_name,
            class_names=class_names,
            class_to_idx=class_to_idx,
            use_amp=device.type == 'cuda',
            config_module=active_config
        )
        
        # Test evaluation
        test_result = evaluation.comprehensive_test_evaluation(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            model_name=model_name,
            config_module=active_config,
            use_amp=device.type == 'cuda'
        )
        
        end_time = time.time()
        
        # Calculate actual dataset sizes
        try:
            train_subset = train_loader.dataset
            test_subset = test_loader.dataset
            train_size_actual = len(train_subset) if isinstance(train_subset, Subset) else 0
            test_size_actual = len(test_subset) if isinstance(test_subset, Subset) else 0
        except:
            train_size_actual = len(train_paths) if 'train_paths' in locals() else 0
            test_size_actual = len(test_paths) if 'test_paths' in locals() else 0
        
        return {
            'model_type': model_type,
            'model_name': model_name,
            'success': True,
            'time': end_time - start_time,
            'test_accuracy': test_result['test_accuracy'],
            'best_val_acc': train_result.get('best_val_acc', 0),
            'train_samples': train_size_actual,
            'labeled_samples': labeled_samples_count,
            'test_samples': test_size_actual,
            'batch_size': active_config.BATCH_SIZE,
            'image_size': active_config.IMAGE_SIZE,
            'epochs': active_config.EPOCHS,
            'few_shot_mode': few_shot_mode,
            'few_shot_value': few_shot_value,
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'model_type': model_type, 'success': False, 'error': str(e), 'time': time.time() - start_time}

def main():
    """Main comparison function"""
    print("🥊 COMPACT MODEL COMPARISON")
    print("="*60)
    
    # 📊 AUTO-DETECT DATASET SIZE
    print("🔍 Getting dataset info...")
    temp_train, temp_labels, temp_val, temp_val_labels, temp_test, temp_test_labels = data_splitter.split_clean_dataset(
        pickle_path='/teamspace/studios/this_studio/thesis_codes/clean_dataset.pkl',
        base_data_dir='/teamspace/studios/this_studio/crop_pest_data',
        few_shot_mode=None  # Just for size detection, actual few-shot applied later
    )
    total_dataset_size = len(temp_train) + len(temp_val) + len(temp_test)
    print(f"📊 Total dataset size: {total_dataset_size:,} samples")
    
    # 📊 DATASET SIZE Configuration:
    SAMPLE_SIZE = None                              # Use full dataset (~25K samples)
    # SAMPLE_SIZE = int(total_dataset_size * 0.1)   # Use 10% of dataset  
    # SAMPLE_SIZE = int(total_dataset_size * 0.01)  # Use 1% of dataset
    # SAMPLE_SIZE = 500                             # Use exactly 500 samples
    
    # 🎯 TRUE FEW-SHOT LEARNING Configuration (LABEL HIDING - not dataset reduction):
    FEW_SHOT_MODE = None                            # Disable few-shot learning
    # FEW_SHOT_MODE = 'percentage'                  # Hide labels: only X% of data has labels
    # FEW_SHOT_MODE = 'per_class'                   # Hide labels: only X samples per class have labels
    FEW_SHOT_VALUE = 0.1                           # 10% labeled data OR 5 samples per class
    
    # 💡 TRUE FEW-SHOT means: Model sees ALL images but most labels are hidden (-1)
    # 💡 This is different from dataset reduction (which would show fewer images)
    
    print(f"🎯 Dataset: {'Full dataset' if SAMPLE_SIZE is None else f'{SAMPLE_SIZE} samples'}")
    if FEW_SHOT_MODE is None:
        print(f"🎯 Few-shot: Disabled (all images have labels)")
    else:
        print(f"🎯 Few-shot: {FEW_SHOT_MODE} ({FEW_SHOT_VALUE}) - LABEL HIDING mode")
        print(f"   💡 Models see ALL images but only some have labels!")
    
    # 🤖 MODEL SELECTION (these are just type labels, real names come from configs)
    model_types = [
        ('cnn', config.MODEL_NAME),        # config.py → e.g., 'efficientnet_b3'
        ('dinov2', config_dinov2.MODEL_NAME)   # config_dinov2.py → e.g., 'dinov2_vits14'
    ]
    
    print(f"🔧 Models to compare:")
    for model_type, actual_name in model_types:
        print(f"   {model_type.upper()}: {actual_name}")
    
    results = []
    
    # 🚀 RUN BOTH MODELS
    for model_type, actual_name in model_types:
        print(f"\n🎯 Running {model_type.upper()} ({actual_name})...")
        result = run_model(model_type, SAMPLE_SIZE, FEW_SHOT_MODE, FEW_SHOT_VALUE)
        results.append(result)
        
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Print comparison
    print("\n" + "="*60)
    print("📊 COMPARISON RESULTS")
    print("="*60)
    
    for result in results:
        if result['success']:
            print(f"\n🏆 {result['model_type'].upper()}: {result['model_name']}")
            print(f"   🎯 Test Accuracy: {result['test_accuracy']:.2f}%")
            print(f"   📈 Best Val: {result['best_val_acc']:.2f}%")
            print(f"   ⏱️  Time: {result['time']:.1f}s")
            
            # Show labeled vs total samples for few-shot
            train_samples = result['train_samples']
            labeled_samples = result.get('labeled_samples', train_samples)
            if result['few_shot_mode'] is not None:
                print(f"   📊 Samples: {train_samples} total train ({labeled_samples} labeled), {result['test_samples']} test")
                print(f"   🎯 Label ratio: {labeled_samples}/{train_samples} ({labeled_samples/train_samples*100:.1f}%)")
            else:
                print(f"   📊 Samples: {train_samples} train, {result['test_samples']} test")
            
            print(f"   batch size: {result['batch_size']}")
            print(f"   image size: {result['image_size']}")
            print(f"   epochs: {result['epochs']}")
            few_shot_info = 'Disabled' if result['few_shot_mode'] is None else f"{result['few_shot_mode']} ({result['few_shot_value']})"
            print(f"   few-shot: {few_shot_info}")
        else:
            print(f"\n💥 {result['model_type'].upper()}: FAILED")
            print(f"   ❌ Error: {result['error']}")
    
    # Winner
    successful = [r for r in results if r['success']]
    if successful:
        best = max(successful, key=lambda x: x['test_accuracy'])
        print(f"\n🏅 WINNER: {best['model_type'].upper()} ({best['test_accuracy']:.2f}%)")

if __name__ == "__main__":
    main() 