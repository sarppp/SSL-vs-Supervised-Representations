#!/usr/bin/env python3
"""
🥊 Compact Model Comparison - CNN vs DINOv2
Super simple 2-model comparison in <100 lines
"""
import torch
import sys
import os
import time
import numpy as np
import random

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the src package to set up all paths
import src

try:
    # Import modules from new structure
    from src.config import config_paths
    from src.data import data_splitter
    from src.data import dataloader_setup
    from src.config import config
    from src.config import config_dinov2
    from src.models import model_setup
    from src.training import training
    from src.evaluation import evaluation
    from src.utils.logger_manager import ComparisonLogger
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

class SimpleConfig:
    """Simple configuration class that can be safely copied"""
    def __init__(self, config_module):
        # Copy all uppercase attributes from the config module
        for attr_name in dir(config_module):
            if attr_name.isupper() and not attr_name.startswith('_'):
                setattr(self, attr_name, getattr(config_module, attr_name))
        
        # Ensure we have the essential attributes (with defaults if missing)
        if not hasattr(self, 'MODEL_NAME'):
            self.MODEL_NAME = getattr(config_module, 'MODEL_NAME', 'unknown_model')
        if not hasattr(self, 'EPOCHS'):
            self.EPOCHS = getattr(config_module, 'EPOCHS', 5)
        if not hasattr(self, 'BATCH_SIZE'):
            self.BATCH_SIZE = getattr(config_module, 'BATCH_SIZE', 16)
        if not hasattr(self, 'LEARNING_RATE'):
            self.LEARNING_RATE = getattr(config_module, 'LEARNING_RATE', 0.001)
        if not hasattr(self, 'IMAGE_SIZE'):
            self.IMAGE_SIZE = getattr(config_module, 'IMAGE_SIZE', (224, 224))
        if not hasattr(self, 'RANDOM_STATE'):
            self.RANDOM_STATE = getattr(config_module, 'RANDOM_STATE', 42)

def run_model(model_type='cnn', sample_size=None, few_shot_mode=None, few_shot_value=0.1, comparison_logger=None):
    """Run single model training and return results"""
    # Select config (create a simple config object that can be safely modified)
    base_config = config_dinov2 if model_type == 'dinov2' else config
    active_config = SimpleConfig(base_config)
    model_name = active_config.MODEL_NAME
    
    # Quick config overrides for fast testing
    active_config.EPOCHS = 2
    active_config.BATCH_SIZE = min(32, active_config.BATCH_SIZE)
    
    # 🔧 OVERRIDE CONFIG FEW-SHOT SETTINGS (prevent other modules from activating few-shot)
    if few_shot_mode is None:
        # Ensure few-shot is completely disabled in config
        setattr(active_config, 'FEW_SHOT_MODE', None)
        setattr(active_config, 'FEW_SHOT_VALUE', None)
    else:
        # Set config few-shot to match our parameters
        setattr(active_config, 'FEW_SHOT_MODE', few_shot_mode)
        setattr(active_config, 'FEW_SHOT_VALUE', few_shot_value)
    
    # Log model start
    if comparison_logger:
        config_info = {
            'epochs': active_config.EPOCHS,
            'batch_size': active_config.BATCH_SIZE,
            'image_size': active_config.IMAGE_SIZE,
            'learning_rate': active_config.LEARNING_RATE,
            'few_shot_mode': few_shot_mode,
            'few_shot_value': few_shot_value
        }
        comparison_logger.log_model_start(model_type, model_name, config_info)
    
    start_time = time.time()
    
    try:
        # Load data (use existing clean dataset) - NO few-shot here, we'll apply label hiding later
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = data_splitter.split_clean_dataset(
            pickle_path=config_paths.CLEAN_DATASET_PICKLE,
            base_data_dir=config_paths.BASE_DATA_DIR,
            few_shot_mode=None  # Don't reduce dataset size
        )
        
        # 📊 LIMIT DATASET SIZE FIRST (before validation!)
        original_train_size = len(train_paths)
        original_val_size = len(val_paths)
        original_test_size = len(test_paths)
        total_original = original_train_size + original_val_size + original_test_size
        
        if sample_size and sample_size < total_original:
            # Calculate limited sizes with better distribution
            min_classes = len(set(train_labels))
            
            # Calculate proportional sizes
            train_size = max(sample_size * 70 // 100, min_classes)
            val_size = max(sample_size * 15 // 100, min_classes // 2)  # At least half the classes
            test_size = max(sample_size * 15 // 100, min_classes // 2)  # At least half the classes
            
            # Ensure we don't exceed sample_size
            total_calculated = train_size + val_size + test_size
            if total_calculated > sample_size:
                # Adjust proportionally while maintaining minimums
                excess = total_calculated - sample_size
                # Remove excess from train first (it's the largest)
                train_size = max(train_size - excess, min_classes)
                
                # Recalculate total
                total_calculated = train_size + val_size + test_size
                if total_calculated > sample_size:
                    # If still over, reduce val and test equally
                    remaining_excess = total_calculated - sample_size
                    val_reduction = remaining_excess // 2
                    test_reduction = remaining_excess - val_reduction
                    val_size = max(val_size - val_reduction, 1)
                    test_size = max(test_size - test_reduction, 1)
            
            # Final safety check - ensure test_size is never 0
            if test_size == 0:
                test_size = min(10, sample_size // 10)  # At least 10 or 10% of sample
                train_size = sample_size - val_size - test_size
            
            # Log dataset processing
            if comparison_logger:
                original_sizes = {'train': original_train_size, 'val': original_val_size, 'test': original_test_size}
                final_sizes = {'train': train_size, 'val': val_size, 'test': test_size}
                comparison_logger.log_dataset_processing(model_type, original_sizes, final_sizes)
            
            # Debug output
            print(f"   📊 Dataset size limiting:")
            print(f"      Original: {original_train_size:,} train, {original_val_size:,} val, {original_test_size:,} test")
            print(f"      Target sample size: {sample_size:,}")
            print(f"      Final: {train_size:,} train, {val_size:,} val, {test_size:,} test")
            print(f"      Total after limiting: {train_size + val_size + test_size:,}")
            
            # Safety check
            if test_size == 0:
                raise ValueError(f"❌ Test size became 0! This is a critical bug. "
                               f"Sample size: {sample_size}, Train: {train_size}, Val: {val_size}")
            
            # Sample train set
            train_indices = np.random.choice(original_train_size, size=train_size, replace=False)
            train_paths = [train_paths[i] for i in train_indices]
            train_labels = [train_labels[i] for i in train_indices]
            
            # Sample val set
            val_indices = np.random.choice(original_val_size, size=val_size, replace=False)
            val_paths = [val_paths[i] for i in val_indices]
            val_labels = [val_labels[i] for i in val_indices]
            
            # Sample test set
            test_indices = np.random.choice(original_test_size, size=test_size, replace=False)
            test_paths = [test_paths[i] for i in test_indices]
            test_labels = [test_labels[i] for i in test_indices]
            
            print(f"   ✅ Pre-sampled {len(train_paths) + len(val_paths) + len(test_paths):,} paths before validation")
        
        # 🎯 TRUE FEW-SHOT: Hide labels instead of reducing dataset size
        labeled_samples_count = len(train_paths)
        
        print(f"🔧 Few-shot check: mode={few_shot_mode}, value={few_shot_value}")
        if few_shot_mode is not None:
            print(f"🎯 COMPACT SCRIPT: Applying few-shot label hiding...")
            train_paths, train_labels, labeled_mask = apply_label_hiding_few_shot(
                train_paths, train_labels, few_shot_mode, few_shot_value, 
                active_config.RANDOM_STATE if hasattr(active_config, 'RANDOM_STATE') else 42
            )
            labeled_samples_count = sum(labeled_mask)
            print(f"🎯 COMPACT SCRIPT: Label Hiding Applied: {labeled_samples_count}/{len(train_paths)} samples have labels")
            few_shot_info = {
                'mode': few_shot_mode,
                'value': few_shot_value,
                'labeled_count': labeled_samples_count,
                'total_count': len(train_paths)
            }
        else:
            print(f"🎯 COMPACT SCRIPT: Few-shot DISABLED - All {len(train_paths)} samples have labels")
        
        # Create dataloaders (now with limited dataset - much faster validation!)
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = dataloader_setup.create_dataloaders(
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
            config_module=active_config, run_batch_test=False
        )
        
        # Get dataset info
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes
        class_to_idx = train_dataset.class_to_idx
        
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
            config_module=active_config,
            test_loader=test_loader  # Pass test_loader for dataset size logging
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
        train_size_actual = len(train_loader.dataset)  # type: ignore
        val_size_actual = len(val_loader.dataset)  # type: ignore
        test_size_actual = len(test_loader.dataset)  # type: ignore
        
        return {
            'model_type': model_type,
            'model_name': model_name,
            'success': True,
            'time': end_time - start_time,
            'train_accuracy': train_result.get('train_accuracies', [0])[-1],  # Get final training accuracy
            'test_accuracy': test_result['test_accuracy'],
            'best_val_acc': train_result.get('best_val_acc', 0),
            'train_samples': train_size_actual,
            'val_samples': val_size_actual,
            'labeled_samples': labeled_samples_count,
            'test_samples': test_size_actual,
            'batch_size': active_config.BATCH_SIZE,
            'image_size': active_config.IMAGE_SIZE,
            'epochs': active_config.EPOCHS,
            'few_shot_mode': few_shot_mode,
            'few_shot_value': few_shot_value,
            'few_shot_info': few_shot_info,
        }
        
    except Exception as e:
        error_result = {'model_type': model_type, 'success': False, 'error': str(e), 'time': time.time() - start_time}
        if comparison_logger:
            comparison_logger.log_model_complete(model_type, model_name, error_result)
        return error_result

def main():
    """Main comparison function"""
    # Initialize comparison logger
    comparison_logger = ComparisonLogger()
    
    # 📊 AUTO-DETECT DATASET SIZE
    temp_train, temp_labels, temp_val, temp_val_labels, temp_test, temp_test_labels = data_splitter.split_clean_dataset(
        pickle_path=config_paths.CLEAN_DATASET_PICKLE,
        base_data_dir=config_paths.BASE_DATA_DIR,
        few_shot_mode=None  # Just for size detection, actual few-shot applied later
    )
    total_dataset_size = len(temp_train) + len(temp_val) + len(temp_test)
    
    # 📊 DATASET SIZE Configuration:
    SAMPLE_SIZE = int(total_dataset_size * 0.1)     # ✅ CURRENTLY ACTIVE: Use 10% of dataset  
    # SAMPLE_SIZE = None                            # Use full dataset (~25K samples)
    # SAMPLE_SIZE = int(total_dataset_size * 0.01)  # Use 1% of dataset
    # SAMPLE_SIZE = 500                             # Use exactly 500 samples
    
    # 🎯 TRUE FEW-SHOT LEARNING Configuration (LABEL HIDING - not dataset reduction):
    #FEW_SHOT_MODE = None                            # Disable few-shot learning
    FEW_SHOT_MODE = 'percentage'                  # Hide labels: only X% of data has labels
    # FEW_SHOT_MODE = 'per_class'                   # Hide labels: only X samples per class have labels
    FEW_SHOT_VALUE = 0.01                           # 10% labeled data OR 5 samples per class
    
    # 💡 TRUE FEW-SHOT means: Model sees ALL images but most labels are hidden (-1)
    # 💡 This is different from dataset reduction (which would show fewer images)
    
    if SAMPLE_SIZE is None:
        print(f"🎯 Dataset: Full dataset ({total_dataset_size:,} samples)")
    else:
        percentage = (SAMPLE_SIZE / total_dataset_size) * 100
        print(f"🎯 Dataset: {SAMPLE_SIZE:,} samples ({percentage:.1f}% of {total_dataset_size:,})")
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
    
    # Log experiment start
    dataset_info = {'total_dataset_size': total_dataset_size}
    experiment_config = {
        'sample_size': SAMPLE_SIZE,
        'few_shot_mode': FEW_SHOT_MODE,
        'few_shot_value': FEW_SHOT_VALUE,
        'models': model_types
    }
    comparison_logger.log_experiment_start(dataset_info, experiment_config)
    
    results = []
    
    # 🚀 RUN BOTH MODELS
    for model_type, actual_name in model_types:
        result = run_model(model_type, SAMPLE_SIZE, FEW_SHOT_MODE, FEW_SHOT_VALUE, comparison_logger)
        results.append(result)
        
        # Log model completion
        comparison_logger.log_model_complete(model_type, actual_name, result)
        
        # Clear GPU cache
        if device.type == 'cuda':
            comparison_logger.log_gpu_cleanup()
            torch.cuda.empty_cache()
    
    # Log comparison results and save
    comparison_logger.log_comparison_results(results)
    comparison_logger.log_experiment_summary()
    results_file = comparison_logger.save_comparison_results(results)

if __name__ == "__main__":
    main() 