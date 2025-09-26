#!/usr/bin/env python3
"""
🥊 Compact Model Comparison - CNN vs DINOv2
Self-contained script for quickly benchmarking two vision backbones.

(NOTE: the original "100 lines" claim is outdated.)
"""
import torch
import sys
import os
import time
import numpy as np
import random

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import modules from new structure
    from src.config import config_paths
    from src.data import data_splitter
    from src.data import dataloader_setup
    from src.config import config
    from src.config import config_dinov2
    from src.config import config_vit
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
torch.cuda.manual_seed(SEED)  # torch.cuda.manual_seed_all is deprecated

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")

def apply_label_hiding_few_shot(train_paths, train_labels, mode='percentage', value=0.1, random_state=42):
    """
    True few-shot learning: Keep ALL training images but hide most labels.
    NOW WITH STRATIFIED SAMPLING FOR FAIR CLASS REPRESENTATION!
    
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
    unique_classes = list(set(train_labels))
    num_classes = len(unique_classes)
    
    print(f"🎯 Few-shot setup: {num_classes} classes, {total_samples} total samples")
    
    if mode == 'percentage':
        # Label only X% of data WITH STRATIFIED SAMPLING
        n_labeled = int(total_samples * value)
        
        # Special case: Allow 0.0 for zero-shot learning (no labeled samples)
        if value == 0.0:
            print(f"🚨 ZERO-SHOT MODE: No labeled samples (experimental)")
        else:
            # Validate: must have at least 1 labeled sample per class for supervised training
            if n_labeled < num_classes and n_labeled > 0:
                raise ValueError(f"❌ FEW_SHOT_VALUE too low: {value} results in {n_labeled} labeled samples, "
                               f"but need at least {num_classes} (1 per class) for training. "
                               f"Use 0.0 for zero-shot or minimum: {num_classes/total_samples:.4f}")
        
        if n_labeled == 0:
            # Zero-shot: no labeled samples
            labeled_indices = np.array([], dtype=int)
        else:
            # 🔥 STRATIFIED SAMPLING: Ensure each class gets fair representation
            labeled_indices = []
            samples_per_class = n_labeled // num_classes
            remaining_samples = n_labeled % num_classes
            
            rng = np.random.RandomState(random_state)
            
            for i, class_name in enumerate(sorted(unique_classes)):
                class_indices = [idx for idx, label in enumerate(train_labels) if label == class_name]
                
                # Give each class at least 'samples_per_class' samples
                n_take = samples_per_class
                
                # Distribute remaining samples to first few classes
                if i < remaining_samples:
                    n_take += 1
                
                # Can't take more samples than available for this class
                n_take = min(n_take, len(class_indices))
                
                if n_take > 0:
                    selected = rng.choice(class_indices, n_take, replace=False)
                    labeled_indices.extend(selected)
            
            labeled_indices = np.array(labeled_indices)
            print(f"🎯 Stratified sampling: {len(labeled_indices)} labeled samples across {num_classes} classes")
        
    elif mode == 'per_class':
        # Label only X samples per class (this is already stratified by design)
        labeled_indices = []
        rng = np.random.RandomState(random_state)
        
        for class_name in sorted(unique_classes):
            class_indices = [i for i, label in enumerate(train_labels) if label == class_name]
            n_take = min(int(value), len(class_indices))
            if n_take > 0:
                selected = rng.choice(class_indices, n_take, replace=False)
                labeled_indices.extend(selected)
        
        labeled_indices = np.array(labeled_indices)
        print(f"🎯 Per-class sampling: {int(value)} samples per class × {num_classes} classes = {len(labeled_indices)} total")
    
    # ------------------------------------------------------------------
    # Efficient O(N) mask creation
    # ------------------------------------------------------------------
    mask = np.zeros(total_samples, dtype=bool)
    mask[labeled_indices] = True

    # Build masked labels in a vectorised manner
    masked_labels = [lbl if m else -1 for lbl, m in zip(train_labels, mask)]
    labeled_mask = mask.tolist()
    
    return train_paths, masked_labels, labeled_mask

def create_few_shot_loss_function(criterion, ignore_index=-1):
    """
    Create loss function that ignores unlabeled samples (-1)
    """
    def few_shot_loss(outputs, targets):
        # Find samples with valid labels (not -1)
        valid_mask = targets != ignore_index
        
        if valid_mask.sum() == 0:
            # No labeled samples in this batch – preserve graph so autograd
            # still keeps references, but gradients will be zero.
            return outputs.sum() * 0.0
        
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

def run_model(model_type='cnn', sample_size=None, few_shot_mode=None, few_shot_value=0.1, 
              experiment_context=None, comparison_logger=None):
    """Run single model training and return results"""
    # Select config (create a simple config object that can be safely modified)
    if model_type == 'dinov2':
        base_config = config_dinov2
    elif model_type == 'vit':
        base_config = config_vit
    else:
        base_config = config
    active_config = SimpleConfig(base_config)
    model_name = active_config.MODEL_NAME
    
    # Research-quality config overrides (not "fast testing"!)
    BATCH_SIZE = 64  # Good for GPU memory
    NUM_WORKERS = 12
    
    # Force override parameters - ensure these take precedence
    active_config.NUM_WORKERS = NUM_WORKERS
    active_config.BATCH_SIZE = BATCH_SIZE
    
    # DON'T override EPOCHS here - let regime-specific function handle it
    
    # Debug: Verify all overrides
    print(f"🔧 Config overrides:")
    print(f"   EPOCHS = {active_config.EPOCHS}")
    print(f"   BATCH_SIZE = {active_config.BATCH_SIZE}")
    print(f"   NUM_WORKERS = {active_config.NUM_WORKERS}")
    # Disable image pre-validation at runtime for faster experiments.  
    # Flip to True if you want to re-run the expensive corruption checks.
    setattr(active_config, 'VALIDATE_IMAGES', False)
    
    # 🔧 OVERRIDE CONFIG FEW-SHOT SETTINGS (prevent other modules from activating few-shot)
    if few_shot_mode is None:
        # Ensure few-shot is completely disabled in config
        setattr(active_config, 'FEW_SHOT_MODE', None)
        setattr(active_config, 'FEW_SHOT_VALUE', None)
    else:
        # Set config few-shot to match our parameters
        setattr(active_config, 'FEW_SHOT_MODE', few_shot_mode)
        setattr(active_config, 'FEW_SHOT_VALUE', few_shot_value)
    
    # Get training_regime from experiment_context (needed for logging and results)
    if experiment_context:
        training_regime_for_log = experiment_context.get('training_regime', 'unknown')
    else:
        training_regime_for_log = 'unknown'
    
    # Log model start with CLEAR EXPERIMENT CONTEXT
    if comparison_logger:
        # Create enhanced config with experiment context
        config_info = {
            'epochs': active_config.EPOCHS,
            'batch_size': active_config.BATCH_SIZE,
            'num_workers': active_config.NUM_WORKERS,
            'image_size': active_config.IMAGE_SIZE,
            'learning_rate': active_config.LEARNING_RATE,
            'weight_decay': getattr(active_config, 'WEIGHT_DECAY', 0.01),
            'dropout': getattr(active_config, 'DROPOUT', 0.1),
            'scheduler': getattr(active_config, 'SCHEDULER', 'plateau'),
            'validate_images': getattr(active_config, 'VALIDATE_IMAGES', False),
            'few_shot_mode': few_shot_mode,
            'few_shot_value': few_shot_value,
            'training_regime': training_regime_for_log,
            'freeze_backbone': getattr(active_config, 'FREEZE_BACKBONE', False),
            'unfreeze_after_epoch': getattr(active_config, 'UNFREEZE_AFTER_EPOCH', None),
        }
        
        # Add experiment context if provided
        if experiment_context:
            config_info.update(experiment_context)
            
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
            
            # Final safety check – ensure neither val_size nor test_size is 0
            if test_size == 0:
                test_size = min(10, sample_size // 10)  # At least 10 or 10% of sample
                train_size = sample_size - val_size - test_size
            
            if val_size == 0:
                val_size = 1
                train_size = max(train_size - 1, min_classes)
            
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
        few_shot_info = None  # Default when few-shot disabled
        
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
            # We still capture basic info for consistency
            few_shot_info = {
                'mode': None,
                'value': None,
                'labeled_count': labeled_samples_count,
                'total_count': len(train_paths)
            }
        
        # Create dataloaders (now with limited dataset - much faster validation!)
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = dataloader_setup.create_dataloaders(
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
            config_module=active_config, run_batch_test=False
        )
        
        # Get dataset info
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes
        class_to_idx = train_dataset.class_to_idx
        
        # Create model (using potentially updated model_name with experiment context)  
        current_model_name = model_name  # Use the updated model_name (which includes experiment context)
        
        # For model creation, use the original architecture name
        if experiment_context:
            # Extract original model name for architecture creation
            original_name = current_model_name.split('_')[0] + '_' + current_model_name.split('_')[1]
            if len(current_model_name.split('_')) > 2:
                # Handle cases like "vit_base_patch16_224_SUPR_10pct"
                parts = current_model_name.split('_')
                if 'vit' in current_model_name:
                    original_name = '_'.join(parts[:4])  # vit_base_patch16_224
                elif 'dinov2' in current_model_name:
                    original_name = '_'.join(parts[:2])  # dinov2_vitb14
                elif 'efficientnet' in current_model_name:
                    original_name = '_'.join(parts[:2])  # efficientnet_b4
        else:
            original_name = current_model_name
            
        model = model_setup.create_model(num_classes, original_name, active_config).to(device)
        
        print(f"🎯 Classes: {num_classes}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
        if experiment_context:
            print(f"💾 Model checkpoints will be saved with identifier: {current_model_name}")
        
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
        
        # Train model (using updated model name with experiment context)
        train_result = training.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_name=current_model_name,  # Use experiment-specific name
            class_names=class_names,
            class_to_idx=class_to_idx,
            use_amp=device.type == 'cuda',
            config_module=active_config,
            test_loader=test_loader  # Pass test_loader for dataset size logging
        )
        
        # Test evaluation (using updated model name with experiment context)
        test_result = evaluation.comprehensive_test_evaluation(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            model_name=current_model_name,  # Use experiment-specific name
            config_module=active_config,
            use_amp=device.type == 'cuda'
        )
        
        end_time = time.time()
        
        # Calculate actual dataset sizes
        train_size_actual = len(train_loader.dataset)  # type: ignore
        val_size_actual = len(val_loader.dataset)  # type: ignore
        test_size_actual = len(test_loader.dataset)  # type: ignore
        
        # Get the best model path from training results
        best_model_path = train_result.get('best_model_path', None)
        
        result = {
            'model_type': model_type,
            'model_name': current_model_name,  # Use experiment-specific name
            'original_model_name': model_name,  # Keep original for reference
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
            'validate_images': getattr(active_config, 'VALIDATE_IMAGES', False),
            'best_model_path': best_model_path,  # Path to saved model
            'training_regime': training_regime_for_log,
            'learning_rate': active_config.LEARNING_RATE,
            'weight_decay': getattr(active_config, 'WEIGHT_DECAY', 0.01),
            'dropout': getattr(active_config, 'DROPOUT', 0.1),
        }
        
        if comparison_logger:
            comparison_logger.log_model_complete(model_type, current_model_name, result)

        return result
        
    except Exception as e:
        current_model_name = getattr(active_config, 'MODEL_NAME', 'unknown')  # Get the updated name if available
        error_result = {'model_type': model_type, 'success': False, 'error': str(e), 'time': time.time() - start_time}
        if comparison_logger:
            comparison_logger.log_model_complete(model_type, current_model_name, error_result)
        return error_result

def run_model_with_regime(model_type='cnn', sample_size=None, few_shot_mode=None, few_shot_value=0.1, 
                         training_regime='supervised', freeze_backbone=False, random_seed=42, 
                         experiment_context=None, comparison_logger=None):
    """Run model with specific training regime (frozen vs fine-tuned)"""
    
    # Select and modify config based on training regime
    if model_type == 'dinov2':
        base_config = config_dinov2
    elif model_type == 'vit':
        base_config = config_vit
    else:
        base_config = config
    
    active_config = SimpleConfig(base_config)
    
    # Get the original model name from config
    original_model_name = active_config.MODEL_NAME
    
    # 🔥 CREATE UNIQUE MODEL IDENTIFIER FOR SAVING
    # Include experiment context in model name so you can distinguish saved models
    if experiment_context:
        label_desc = experiment_context['label_description']
        regime_short = training_regime.replace('_', '').upper()[:4]  # e.g., LINP, FINE, SUPR
        seed_str = f"_s{experiment_context['random_seed']}" if experiment_context['random_seed'] != 42 else ""
        
        # Create unique model identifier: e.g., "dinov2_vitb14_FINE_50pct_s42"
        unique_model_id = f"{original_model_name}_{regime_short}_{label_desc.replace('%', 'pct')}{seed_str}"
        
        # Override the model name in config so all saving uses this unique identifier
        active_config.MODEL_NAME = unique_model_id
        
        # Set the model name for this function
        model_name = unique_model_id
        
        print(f"📝 Model will be saved as: {unique_model_id}")
        print(f"📝 All logs will use identifier: {unique_model_id}")
    else:
        model_name = original_model_name
    
    # 🔥 FAIR COMPETITION: Research-quality hyperparameters optimized for each strategy
    if training_regime == 'linear_probe':
        # Linear probe: freeze backbone, only train classifier head
        active_config.FREEZE_BACKBONE = True
        active_config.UNFREEZE_AFTER_EPOCH = 999  # Never unfreeze
        active_config.EPOCHS = 5   # ✅ QUICK TEST: Reduced epochs (was 15)
        active_config.LEARNING_RATE = 0.001  # Fair LR for all models in linear probe
        active_config.WEIGHT_DECAY = 0.01
        
        # Model-specific dropout (pre-trained features need less regularization)
        if model_type == 'dinov2':
            active_config.DROPOUT = 0.1  # Lower dropout for pre-trained features
        else:
            active_config.DROPOUT = 0.15  # Slightly higher for CNN/ViT linear probe
            
        print(f"🧊 Linear probe: frozen backbone, {active_config.EPOCHS} epochs, LR={active_config.LEARNING_RATE}")
        
    elif training_regime == 'fine_tune':
        # Fine-tuning: progressive unfreezing (only for DiNO, but keeping general)
        active_config.FREEZE_BACKBONE = True
        active_config.UNFREEZE_AFTER_EPOCH = 3   # ✅ QUICK TEST: Earlier unfreeze (was 5)
        active_config.EPOCHS = 8   # ✅ QUICK TEST: Reduced epochs (was 25)
        active_config.LEARNING_RATE = 0.0005  # Lower LR for stable fine-tuning
        active_config.WEIGHT_DECAY = 0.01
        active_config.DROPOUT = 0.1  # Lower dropout for fine-tuning
            
        print(f"🔥 Fine-tune: progressive unfreeze @ epoch {active_config.UNFREEZE_AFTER_EPOCH}, {active_config.EPOCHS} epochs, LR={active_config.LEARNING_RATE}")
        
    else:  # supervised
        # Supervised: full training from scratch
        active_config.FREEZE_BACKBONE = False
        active_config.EPOCHS = 8   # ✅ QUICK TEST: Reduced epochs (was 25)
        active_config.WEIGHT_DECAY = 0.01
        
        # Model-specific learning rates (fair but optimized)
        if model_type == 'cnn':
            active_config.LEARNING_RATE = 0.001   # Standard for CNNs
            active_config.DROPOUT = 0.2           # Higher dropout for training from scratch
        elif model_type == 'vit':
            active_config.LEARNING_RATE = 0.0008  # Slightly lower for ViTs
            active_config.DROPOUT = 0.1           # ViT-appropriate dropout
        else:  # dinov2 supervised (theoretical case)
            active_config.LEARNING_RATE = 0.0005  # Conservative for ViT architecture
            active_config.DROPOUT = 0.1
            
        print(f"🚀 Supervised: full training, {active_config.EPOCHS} epochs, LR={active_config.LEARNING_RATE}")
    
    # 🎯 TRAINING STRATEGY OPTIMIZATIONS
    
    # Learning rate scheduling (regime-appropriate)
    if training_regime == 'linear_probe':
        # Linear probe: step decay works well for frozen features
        active_config.SCHEDULER = 'step'
        active_config.SCHEDULER_PARAMS = {'step': {'step_size': 7, 'gamma': 0.5}}
        active_config.USE_WARMUP = False  # No warmup needed for linear probe
        
    elif training_regime == 'fine_tune':
        # Fine-tuning: cosine annealing for smooth feature adaptation
        active_config.SCHEDULER = 'cosine'
        active_config.SCHEDULER_PARAMS = {'cosine': {'T_max': active_config.EPOCHS, 'eta_min': 1e-7}}
        active_config.USE_WARMUP = True
        active_config.WARMUP_EPOCHS = 3
        active_config.WARMUP_START_LR = 1e-7
        
    else:  # supervised
        # Supervised: plateau scheduler for adaptive learning
        active_config.SCHEDULER = 'plateau'
        active_config.SCHEDULER_PARAMS = {'plateau': {'mode': 'min', 'patience': 5, 'factor': 0.5}}
        active_config.USE_WARMUP = True
        active_config.WARMUP_EPOCHS = 3
        active_config.WARMUP_START_LR = 1e-6
    
    # Override random seed for reproducibility
    active_config.RANDOM_STATE = random_seed
    
    # Apply the modified config to the current context
    # Store original config and replace with modified one
    if model_type == 'dinov2':
        original_config = config_dinov2
        # Temporarily replace the module attributes
        for attr in ['FREEZE_BACKBONE', 'UNFREEZE_AFTER_EPOCH', 'LEARNING_RATE', 'RANDOM_STATE', 
                     'EPOCHS', 'WEIGHT_DECAY', 'DROPOUT', 'SCHEDULER', 'SCHEDULER_PARAMS', 
                     'USE_WARMUP', 'WARMUP_EPOCHS', 'WARMUP_START_LR']:
            if hasattr(active_config, attr):
                setattr(original_config, attr, getattr(active_config, attr))
    elif model_type == 'vit':
        original_config = config_vit
        for attr in ['FREEZE_BACKBONE', 'UNFREEZE_AFTER_EPOCH', 'LEARNING_RATE', 'RANDOM_STATE',
                     'EPOCHS', 'WEIGHT_DECAY', 'DROPOUT', 'SCHEDULER', 'SCHEDULER_PARAMS',
                     'USE_WARMUP', 'WARMUP_EPOCHS', 'WARMUP_START_LR']:
            if hasattr(active_config, attr):
                setattr(original_config, attr, getattr(active_config, attr))
    else:
        original_config = config
        for attr in ['FREEZE_BACKBONE', 'UNFREEZE_AFTER_EPOCH', 'LEARNING_RATE', 'RANDOM_STATE',
                     'EPOCHS', 'WEIGHT_DECAY', 'DROPOUT', 'SCHEDULER', 'SCHEDULER_PARAMS',
                     'USE_WARMUP', 'WARMUP_EPOCHS', 'WARMUP_START_LR']:
            if hasattr(active_config, attr):
                setattr(original_config, attr, getattr(active_config, attr))
    
    # Use the existing run_model function
    return run_model(
        model_type=model_type,
        sample_size=sample_size,
        few_shot_mode=few_shot_mode,
        few_shot_value=few_shot_value,
        experiment_context=experiment_context,
        comparison_logger=comparison_logger
    )

def aggregate_results_by_condition(all_results):
    """Aggregate results by (budget, regime, model) to compute mean ± std"""
    from collections import defaultdict
    import statistics
    
    # Group results by condition
    grouped = defaultdict(list)
    
    for result in all_results:
        if not result.get('success', False):
            continue  # Skip failed experiments
            
        # Create condition key
        condition = (
            result['budget_mode'],
            result['budget_value'], 
            result['model_type'],
            result['training_regime']
        )
        
        grouped[condition].append(result['test_accuracy'])
    
    # Compute statistics for each condition
    aggregated = {}
    for condition, accuracies in grouped.items():
        if len(accuracies) > 0:
            mean_acc = statistics.mean(accuracies)
            std_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
            
            budget_mode, budget_value, model_type, regime = condition
            aggregated[condition] = {
                'budget_mode': budget_mode,
                'budget_value': budget_value,
                'model_type': model_type,
                'training_regime': regime,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'num_runs': len(accuracies),
                'all_accuracies': accuracies
            }
    
    return aggregated

def print_statistical_summary(aggregated_results):
    """Print statistical summary of results"""
    print("\n" + "="*80)
    print("📊 STATISTICAL SUMMARY (Mean ± Std)")
    print("="*80)
    
    # Sort by budget value for easy comparison
    sorted_results = sorted(aggregated_results.items(), 
                          key=lambda x: (x[1]['budget_value'], x[1]['model_type'], x[1]['training_regime']))
    
    current_budget = None
    for condition, stats in sorted_results:
        budget_str = f"{stats['budget_mode']}={stats['budget_value']}"
        
        if budget_str != current_budget:
            print(f"\n🎯 Label Budget: {budget_str}")
            print("-" * 50)
            current_budget = budget_str
        
        regime_str = f"{stats['training_regime']}" if stats['training_regime'] != 'supervised' else ""
        model_name = f"{stats['model_type'].upper()}"
        if regime_str:
            model_name += f" ({regime_str})"
        
        print(f"{model_name:25} {stats['mean_accuracy']:6.2f}% ± {stats['std_accuracy']:5.2f}% (n={stats['num_runs']})")
    
    print("\n" + "="*80)

def main():
    """Main comparison function with MULTIPLE SEEDS and TRAINING REGIMES"""
    print("🚀 QUICK TEST MODE: 1000 samples, reduced epochs (~1-2 hours)")
    print("🔧 For full experiment: change SAMPLE_SIZE=10000, EPOCHS back to (15,25,25)")
    
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
    # SAMPLE_SIZE = 10000                            # Full experiment (9-13 hours)
    # SAMPLE_SIZE = 1000                             # ✅ QUICK TEST: Small dataset (~1-2 hours)
    SAMPLE_SIZE = 500                             # Even faster test
    
    # 🎯 CRITICAL: Label efficiency analysis - how much labeled data is needed?
    LABEL_BUDGETS = [
        ('percentage', 0.1),   # 10% of labels visible (few-shot learning)
        ('percentage', 0.5),   # 50% of labels visible (medium-shot learning)  
        ('percentage', 1.0)    # 100% of labels visible (full supervision)
    ]
    
    # 🎯 Single seed for faster experimentation (can expand to 3+ seeds later for publication)
    RANDOM_SEEDS = [42]  # Single seed for manageable runtime
    
    # 🎯 CRITICAL: Different training regimes for fair comparison
    TRAINING_REGIMES = {
        'cnn': ['supervised'],  # CNN always supervised
        'dinov2': ['linear_probe', 'fine_tune'],  # DINOv2: frozen vs fine-tuned
        'vit': ['supervised']   # ViT supervised (can add frozen if needed)
    }
    
    print(f"🎯 Dataset: {SAMPLE_SIZE:,} samples ({SAMPLE_SIZE/total_dataset_size*100:.1f}% of {total_dataset_size:,})")
    print(f"🎯 Label budgets: {len(LABEL_BUDGETS)} budgets × {len(RANDOM_SEEDS)} seeds × models")
    print(f"🎯 Training regimes: {TRAINING_REGIMES}")
    
    # 🤖 MODEL SELECTION 
    model_types = [
        ('cnn', config.MODEL_NAME),
        ('dinov2', config_dinov2.MODEL_NAME),
        ('vit', config_vit.MODEL_NAME)
    ]
    
    # Log experiment start
    dataset_info = {'total_dataset_size': total_dataset_size}
    experiment_config = {
        'sample_size': SAMPLE_SIZE,
        'label_budgets': LABEL_BUDGETS,
        'random_seeds': RANDOM_SEEDS,
        'training_regimes': TRAINING_REGIMES,
        'models': model_types
    }
    comparison_logger.log_experiment_start(dataset_info, experiment_config)
    
    all_results = []
    
    # 🚀 RUN COMPREHENSIVE EXPERIMENTS
    total_experiments = len(LABEL_BUDGETS) * len(RANDOM_SEEDS) * sum(len(regimes) for regimes in TRAINING_REGIMES.values())
    experiment_count = 0
    
    for budget_idx, (budget_mode, budget_value) in enumerate(LABEL_BUDGETS, 1):
        label_desc = f"{int(budget_value*100)}%" if budget_mode == 'percentage' else f"{budget_value}/class"
        
        print(f"\n" + "="*80)
        print(f"🔥 LABEL BUDGET {budget_idx}/{len(LABEL_BUDGETS)}: {label_desc} of training labels visible")
        print(f"   📊 Mode: {budget_mode} | Value: {budget_value}")
        print("="*80)
        
        for seed in RANDOM_SEEDS:
            print(f"\n🎲 Random seed: {seed}")
            
            # Update global seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            for model_type, actual_name in model_types:
                regimes = TRAINING_REGIMES[model_type]
                
                for regime in regimes:
                    experiment_count += 1
                    
                    # 🔥 ENHANCED EXPERIMENT IDENTIFICATION
                    label_desc = f"{int(budget_value*100)}%" if budget_mode == 'percentage' else f"{budget_value}/class"
                    experiment_title = f"{model_type.upper()}-{regime.upper()}, {label_desc} labels"
                    
                    print(f"\n🚀 EXPERIMENT [{experiment_count}/{total_experiments}]: {experiment_title}")
                    print(f"   🎲 Seed: {seed} | 🏷️ Budget: {budget_mode}={budget_value}")
                    
                    # Create experiment context for logging
                    experiment_context = {
                        'experiment_number': experiment_count,
                        'total_experiments': total_experiments,
                        'experiment_title': experiment_title,
                        'label_description': label_desc,
                        'budget_mode': budget_mode,
                        'budget_value': budget_value,
                        'random_seed': seed,
                        'training_regime': regime  # Add training regime to context
                    }
                    
                    # Modify config based on training regime
                    freeze_backbone = (regime == 'linear_probe')
                    
                    result = run_model_with_regime(
                        model_type=model_type,
                        sample_size=SAMPLE_SIZE,
                        few_shot_mode=budget_mode,
                        few_shot_value=budget_value,
                        training_regime=regime,
                        freeze_backbone=freeze_backbone,
                        random_seed=seed,
                        experiment_context=experiment_context,
                        comparison_logger=comparison_logger
                    )
                    
                    # Add experiment metadata  
                    result.update({
                        'budget_mode': budget_mode,
                        'budget_value': budget_value,
                        'training_regime': regime,
                        'random_seed': seed,
                        'freeze_backbone': freeze_backbone,
                        'experiment_number': experiment_count,
                        'experiment_title': experiment_title
                    })
                    
                    all_results.append(result)
                    
                    # 📊 EXPERIMENT COMPLETION SUMMARY
                    if result.get('success', False):
                        test_acc = result.get('test_accuracy', 0)
                        train_acc = result.get('train_accuracy', 0)
                        model_path = result.get('best_model_path', 'No path available')
                        print(f"   ✅ COMPLETED: Train={train_acc:.1f}%, Test={test_acc:.1f}%")
                        if model_path and model_path != 'No path available':
                            print(f"   💾 Model saved: {model_path}")
                        else:
                            print(f"   💾 Model saved as: {result.get('model_name', 'unknown')}")
                    else:
                        print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
                    
                    print(f"   📈 Progress: {experiment_count}/{total_experiments} experiments done")
                    print("   " + "="*60)
                    
                    # Clear GPU cache
                    if device.type == 'cuda':
                        comparison_logger.log_gpu_cleanup()
                        torch.cuda.empty_cache()
    
    # 📊 AGGREGATE RESULTS: Compute mean ± std per (budget, regime, model)
    aggregated_results = aggregate_results_by_condition(all_results)
    
    # Log and save results
    comparison_logger.log_comparison_results(all_results)
    comparison_logger.log_experiment_summary()
    results_file = comparison_logger.save_comparison_results(all_results)
    
    # Save aggregated results separately (in case logger doesn't support it)
    try:
        import json
        import os
        aggregated_file = results_file.replace('.json', '_aggregated.json')
        with open(aggregated_file, 'w') as f:
            # Convert condition tuples to strings for JSON serialization
            json_compatible = {}
            for condition, stats in aggregated_results.items():
                key = f"{stats['model_type']}_{stats['training_regime']}_{stats['budget_mode']}{stats['budget_value']}"
                json_compatible[key] = stats
            json.dump(json_compatible, f, indent=2)
        print(f"📊 Aggregated results saved to: {aggregated_file}")
    except Exception as e:
        print(f"⚠️ Could not save aggregated results: {e}")
    
    # 🎉 FINAL EXPERIMENT SUMMARY
    print(f"\n" + "="*80)
    print(f"🎉 COMPREHENSIVE EXPERIMENT COMPLETED!")
    print(f"📊 Total experiments run: {len(all_results)}")
    print(f"✅ Successful: {sum(1 for r in all_results if r.get('success', False))}")
    print(f"❌ Failed: {sum(1 for r in all_results if not r.get('success', False))}")
    print(f"📁 Results saved to: {results_file}")
    print(f"📁 Aggregated results: {aggregated_file}")
    
    # 💾 SAVED MODELS SUMMARY
    print(f"\n💾 SAVED MODELS SUMMARY:")
    print("-" * 80)
    successful_results = [r for r in all_results if r.get('success', False)]
    if successful_results:
        for result in successful_results:
            experiment_title = result.get('experiment_title', 'Unknown')
            model_name = result.get('model_name', 'Unknown')
            test_acc = result.get('test_accuracy', 0)
            model_path = result.get('best_model_path', 'No path')
            
            print(f"📌 {experiment_title}: {test_acc:.1f}%")
            print(f"   🔧 Model ID: {model_name}")
            if model_path and model_path != 'No path':
                print(f"   📁 Path: {model_path}")
            print()
    else:
        print("   No models were successfully saved.")
    
    print("="*80)
    
    print_statistical_summary(aggregated_results)

if __name__ == "__main__":
    main() 