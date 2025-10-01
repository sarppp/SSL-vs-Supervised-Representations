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
    from src.utils.logger_manager import ComparisonLogger, DataSplitterLogger
    from src.models.model_setup import get_model_identifier
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# 🔧 OPTIONAL: GPU Configuration presets (comment out if not using)
try:
    from gpu_configs import L40S_CONSERVATIVE, A100_OPTIMAL, A100_ULTRA, H100_OPTIMAL, H100_ULTRA
    GPU_CONFIGS_AVAILABLE = True
    print("✅ GPU configs imported - You can use preset configurations")
except ImportError:
    GPU_CONFIGS_AVAILABLE = False
    print("ℹ️  GPU configs not found - Using manual configuration")

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
    # Defensive cast: ensure numeric value for comparisons and arithmetic
    try:
        value = float(value)
    except Exception:
        value = 0.0
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
            
            for i, class_name in enumerate(sorted(unique_classes, key=str)):
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
        
        for class_name in sorted(unique_classes, key=str):
            class_indices = [i for i, label in enumerate(train_labels) if label == class_name]
            try:
                per_class_n = int(value)
            except Exception:
                per_class_n = 0
            n_take = min(per_class_n, len(class_indices))
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
        # GPU-SPECIFIC SETTINGS:
    # - A100 80GB: Use PER_STEP_BATCH=256, GRADIENT_ACCUM_STEPS=1 (no accumulation needed!)
    # - L40S 24GB: Use PER_STEP_BATCH=128, GRADIENT_ACCUM_STEPS=2 (effective 256)
    # 🚀 OPTIMIZED CONFIG: Large-scale training with 25k samples
    # 
    # 🎯 QUICK GPU SWITCHING: Uncomment the preset you want to use
    # Choose ONE option below:
    
    # OPTION 1A: L40S 24GB (Safe) - ~7-8 hours
    GPU_CONFIG = L40S_CONSERVATIVE if GPU_CONFIGS_AVAILABLE else None
    
    # OPTION 1B: A100 80GB (Optimal) - ~3-4 hours ⚡
    # GPU_CONFIG = A100_OPTIMAL if GPU_CONFIGS_AVAILABLE else None
    
    # OPTION 1C: A100 80GB (Ultra-Fast) - ~2-3 hours 
    # GPU_CONFIG = A100_ULTRA if GPU_CONFIGS_AVAILABLE else None

    # OPTION 1D: H100 80GB (Optimal) - ~2-3 hours ⚡
    # GPU_CONFIG = H100_OPTIMAL if GPU_CONFIGS_AVAILABLE else None

    # OPTION 1E: H100 80GB (Ultra-Fast) - ~1.5-2.5 hours 🚀
    # GPU_CONFIG = H100_ULTRA if GPU_CONFIGS_AVAILABLE else None
    
    if GPU_CONFIG is not None:
        PER_STEP_BATCH = GPU_CONFIG['per_step_batch']
        GRADIENT_ACCUM_STEPS = GPU_CONFIG['gradient_accum_steps']
        EFFECTIVE_BATCH = GPU_CONFIG['effective_batch']
        NUM_WORKERS = GPU_CONFIG['num_workers']
        print(f"🔧 Using preset: {GPU_CONFIG['name']}")
        print(f"📊 {GPU_CONFIG['description']}")
        # Auto-scale learning rate when the preset declares it
        if GPU_CONFIG.get('auto_scale_lr', False):
            base_lr = active_config.LEARNING_RATE
            lr_multiplier = GPU_CONFIG.get('lr_multiplier')
            if lr_multiplier is not None:
                active_config.LEARNING_RATE = base_lr * float(lr_multiplier)
                try:
                    print(f"⚙️  Auto LR scaling: {base_lr:.6g} × {float(lr_multiplier):.3g} -> {active_config.LEARNING_RATE:.6g}")
                except Exception:
                    print(f"⚙️  Auto LR scaling applied. New LR = {active_config.LEARNING_RATE}")
            else:
                # Fallback to batch-size-based scaling via helper if available
                try:
                    from gpu_configs import get_scaled_lr
                    active_config.LEARNING_RATE = get_scaled_lr(
                        base_lr,
                        64,
                        EFFECTIVE_BATCH,
                        'linear'
                    )
                    print(f"⚙️  Auto LR scaling (by batch): {base_lr:.6g} -> {active_config.LEARNING_RATE:.6g}")
                except Exception:
                    print("⚠️  Could not auto-scale LR; using base learning rate.")
    else:
        # Fallback: Manual configuration
        PER_STEP_BATCH = 128      # Physical batch per step (safe for L40S 24GB)
        GRADIENT_ACCUM_STEPS = 2  # Accumulate 2 steps (effective batch 256)
        EFFECTIVE_BATCH = 256     # Effective batch = 128 × 2 = 256
        NUM_WORKERS = 8           # Optimal for most systems
        print("ℹ️  Using manual GPU configuration")
    
    # Force override parameters - ensure these take precedence
    active_config.NUM_WORKERS = NUM_WORKERS
    active_config.BATCH_SIZE = PER_STEP_BATCH  # Physical batch size
    active_config.GRADIENT_ACCUM_STEPS = GRADIENT_ACCUM_STEPS
    active_config.EFFECTIVE_BATCH_SIZE = EFFECTIVE_BATCH
    
    # DON'T override EPOCHS here - let regime-specific function handle it
    
    # Debug: Verify all overrides
    print(f"🔧 Config overrides:")
    print(f"   EPOCHS = {active_config.EPOCHS}")
    print(f"   PER-STEP BATCH = {active_config.BATCH_SIZE}")
    print(f"   GRADIENT ACCUM STEPS = {active_config.GRADIENT_ACCUM_STEPS}")
    print(f"   EFFECTIVE BATCH = {active_config.EFFECTIVE_BATCH_SIZE}")
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
        # Load data using config-based splitter to get per-model data splitter logs
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = data_splitter.split_clean_dataset_with_config(
            pickle_path=config_paths.CLEAN_DATASET_PICKLE,
            config_module=active_config,
            base_data_dir=config_paths.BASE_DATA_DIR
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
            
            # 🔧 FIX: Stratified sampling to ensure ALL classes are represented
            # (prevents KeyError when small samples miss some classes)
            
            def stratified_sample(paths, labels, target_size, random_state=42):
                """Sample while ensuring all classes are represented"""
                from collections import defaultdict
                
                # Group indices by class
                class_indices = defaultdict(list)
                for idx, label in enumerate(labels):
                    class_indices[label].append(idx)
                
                unique_classes = list(class_indices.keys())
                n_classes = len(unique_classes)
                
                # Calculate samples per class (ensure at least 1 per class)
                samples_per_class = max(1, target_size // n_classes)
                
                rng = np.random.RandomState(random_state)
                sampled_indices = []
                
                for class_label in unique_classes:
                    indices = class_indices[class_label]
                    n_take = min(samples_per_class, len(indices))
                    selected = rng.choice(indices, n_take, replace=False)
                    sampled_indices.extend(selected)
                
                # If we haven't reached target_size, add more samples
                if len(sampled_indices) < target_size:
                    remaining = target_size - len(sampled_indices)
                    all_indices = list(range(len(paths)))
                    unused = [i for i in all_indices if i not in sampled_indices]
                    if unused:
                        extra = rng.choice(unused, min(remaining, len(unused)), replace=False)
                        sampled_indices.extend(extra)
                
                # Sample paths and labels
                sampled_paths = [paths[i] for i in sampled_indices]
                sampled_labels = [labels[i] for i in sampled_indices]
                return sampled_paths, sampled_labels
            
            # Apply stratified sampling to each split
            train_paths, train_labels = stratified_sample(train_paths, train_labels, train_size)
            val_paths, val_labels = stratified_sample(val_paths, val_labels, val_size)
            test_paths, test_labels = stratified_sample(test_paths, test_labels, test_size)
            
            print(f"   ✅ Pre-sampled {len(train_paths) + len(val_paths) + len(test_paths):,} paths before validation")
        else:
            # Using full dataset
            print(f"   📊 FULL DATASET MODE:")
            print(f"      Train: {original_train_size:,} samples")
            print(f"      Val: {original_val_size:,} samples")
            print(f"      Test: {original_test_size:,} samples")
            print(f"      Total: {total_original:,} samples")
            print(f"   ✅ Using complete dataset for research-quality training")
        
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
        
        # Create model using the original architecture name (not the full identifier)
        # The full identifier will be generated by get_model_identifier() in training
        original_name = model_name  # This should be the base model name like 'dinov2_vitb14'
        
        # Generate the full model identifier for logging and saving
        from src.models.model_setup import get_model_identifier
        current_model_name = get_model_identifier(original_name, active_config)
            
        model = model_setup.create_model(num_classes, original_name, active_config).to(device)
        
        print(f"🎯 Classes: {num_classes}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
        if experiment_context:
            print(f"💾 Model checkpoints will be saved with identifier: {current_model_name}")
        
        # Create optimizer and scheduler (ViT-friendly: AdamW + warmup + cosine)
        # Build ViT param groups with proper weight-decay exclusions
        param_groups = None
        try:
            if isinstance(original_name, str) and original_name.startswith('vit_'):
                no_decay_keywords = ['bias', 'pos_embed', 'cls_token', 'norm']
                decay_params = []
                no_decay_params = []
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if any(k in name for k in no_decay_keywords):
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)
                if len(decay_params) > 0 or len(no_decay_params) > 0:
                    wd = getattr(active_config, 'WEIGHT_DECAY', 0.05)
                    param_groups = [
                        {'params': decay_params, 'weight_decay': wd},
                        {'params': no_decay_params, 'weight_decay': 0.0},
                    ]
        except Exception:
            param_groups = None

        optimizer = model_setup.create_optimizer(param_groups if param_groups is not None else model.parameters(), config=active_config)
        scheduler = model_setup.create_scheduler(optimizer, config=active_config)
        
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
        
        # Compute throughput (images/sec) and record GPU preset info when available
        total_train_images = train_size_actual
        total_time_sec = end_time - start_time
        images_per_sec = (total_train_images * active_config.EPOCHS) / total_time_sec if total_time_sec > 0 else None
        gpu_preset_name = GPU_CONFIG['name'] if 'GPU_CONFIG' in locals() and GPU_CONFIG is not None else 'manual'
        
        result = {
            'model_type': model_type,
            'model_name': current_model_name,  # Use experiment-specific name
            'original_model_name': model_name,  # Keep original for reference
            'success': True,
            'time': end_time - start_time,
            'time_per_epoch': (end_time - start_time) / active_config.EPOCHS if active_config.EPOCHS else None,
            'images_per_sec': images_per_sec,
            'gpu_preset': gpu_preset_name,
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
            # Include evaluation metrics for thesis plots/tables if available
            'precision_macro': float(test_result.get('precision_macro')) if test_result.get('precision_macro') is not None else None,
            'recall_macro': float(test_result.get('recall_macro')) if test_result.get('recall_macro') is not None else None,
            'f1_macro': float(test_result.get('f1_macro')) if test_result.get('f1_macro') is not None else None,
            'matthews_corrcoef': float(test_result.get('matthews_corrcoef')) if test_result.get('matthews_corrcoef') is not None else None,
            'roc_auc_ovr': float(test_result.get('roc_auc_ovr')) if test_result.get('roc_auc_ovr') is not None else None,
        }
        
        if comparison_logger:
            comparison_logger.log_model_complete(model_type, current_model_name, result)

        return result
        
    except Exception as e:
        # Generate proper model identifier for error logging
        try:
            from src.models.model_setup import get_model_identifier
            current_model_name = get_model_identifier(model_name, active_config)
        except:
            current_model_name = getattr(active_config, 'MODEL_NAME', 'unknown')
        
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
    
    # ✅ Set regime and label budget on config; DO NOT override MODEL_NAME
    # Let get_model_identifier() derive consistent IDs across the codebase
    if experiment_context:
        label_desc = experiment_context['label_description']
        seed_str = f"_s{experiment_context['random_seed']}" if experiment_context['random_seed'] != 42 else ""
        print(f"📝 Experiment context: {training_regime} | {label_desc}{seed_str}")
    # Ensure training regime is available to identifier and logs
    active_config.TRAINING_REGIME = training_regime
    # Keep using original model name; identifier will include regime and labels
    model_name = original_model_name
    
    # 🔥 FAIR COMPETITION: Research-quality hyperparameters optimized for each strategy
    # 📈 LEARNING RATE SCALING: Adjusted for effective batch 256 (4× from base batch 64)
    if training_regime == 'linear_probe':
        # Linear probe: freeze backbone, only train classifier head
        active_config.FREEZE_BACKBONE = True
        active_config.UNFREEZE_AFTER_EPOCH = int(999)  # Never unfreeze
        active_config.EPOCHS = int(5)  # Linear probe: quick convergence (was 5 for testing) #15
        active_config.LEARNING_RATE = float(0.004)  # 0.001 × 4 (linear scaling for batch 256)
        active_config.WEIGHT_DECAY = float(0.01)
        
        # Model-specific dropout (pre-trained features need less regularization)
        if model_type == 'dinov2':
            active_config.DROPOUT = float(0.1)  # Lower dropout for pre-trained features
        else:
            active_config.DROPOUT = float(0.15)  # Slightly higher for CNN/ViT linear probe
            
        print(f"🧊 Linear probe: frozen backbone, {active_config.EPOCHS} epochs, LR={active_config.LEARNING_RATE}")
        
    elif training_regime == 'fine_tune':
        # Fine-tuning: progressive unfreezing (only for DiNO, but keeping general)
        active_config.FREEZE_BACKBONE = True
        active_config.UNFREEZE_AFTER_EPOCH = int(3)   # ✅ Early unfreeze
        active_config.EPOCHS = int(8)  # Fine-tuning: needs more epochs (was 8 for testing) 25
        active_config.LEARNING_RATE = float(0.002)  # 0.0005 × 4 (linear scaling for batch 256)
        active_config.WEIGHT_DECAY = float(0.01)
        active_config.DROPOUT = float(0.1)  # Lower dropout for fine-tuning
            
        print(f"🔥 Fine-tune: progressive unfreeze @ epoch {active_config.UNFREEZE_AFTER_EPOCH}, {active_config.EPOCHS} epochs, LR={active_config.LEARNING_RATE}")
        
    else:  # supervised
        # Supervised: full training from scratch
        active_config.FREEZE_BACKBONE = False
        active_config.EPOCHS = int(8)  # Supervised: full training from scratch 25
        active_config.WEIGHT_DECAY = float(0.01)
        
        # Model-specific learning rates (scaled for batch 256)
        if model_type == 'cnn':
            active_config.LEARNING_RATE = float(0.001)   # 0.001 × 4 (linear scaling)
            active_config.DROPOUT = float(0.2)           # Higher dropout for training from scratch
        elif model_type == 'vit':
            active_config.LEARNING_RATE = float(0.0005)  # Optimized LR for ViT stability
            active_config.DROPOUT = float(0.15)          # Higher dropout for better generalization
        else:  # dinov2 supervised (theoretical case)
            active_config.LEARNING_RATE = float(0.0005)   # 0.0005 × 4 (linear scaling)
            active_config.DROPOUT = float(0.1)
            
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
        # Supervised: cosine scheduler for ViT, plateau for others
        if model_type == 'vit':
            active_config.SCHEDULER = 'cosine'
            active_config.SCHEDULER_PARAMS = {'cosine': {'T_max': active_config.EPOCHS, 'eta_min': 1e-6}}
        else:
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
        for attr in ['FREEZE_BACKBONE', 'UNFREEZE_AFTER_EPOCH', 'LEARNING_RATE', 'RANDOM_STATE', 'TRAINING_REGIME',
                     'EPOCHS', 'WEIGHT_DECAY', 'DROPOUT', 'SCHEDULER', 'SCHEDULER_PARAMS', 
                     'USE_WARMUP', 'WARMUP_EPOCHS', 'WARMUP_START_LR']:
            if hasattr(active_config, attr):
                setattr(original_config, attr, getattr(active_config, attr))
    elif model_type == 'vit':
        original_config = config_vit
        for attr in ['FREEZE_BACKBONE', 'UNFREEZE_AFTER_EPOCH', 'LEARNING_RATE', 'RANDOM_STATE', 'TRAINING_REGIME',
                     'EPOCHS', 'WEIGHT_DECAY', 'DROPOUT', 'SCHEDULER', 'SCHEDULER_PARAMS',
                     'USE_WARMUP', 'WARMUP_EPOCHS', 'WARMUP_START_LR']:
            if hasattr(active_config, attr):
                setattr(original_config, attr, getattr(active_config, attr))
    else:
        original_config = config
        for attr in ['FREEZE_BACKBONE', 'UNFREEZE_AFTER_EPOCH', 'LEARNING_RATE', 'RANDOM_STATE', 'TRAINING_REGIME',
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
    def sort_key(item):
        stats = item[1]
        budget_val = stats.get('budget_value')
        try:
            budget_val_num = float(budget_val)
        except Exception:
            budget_val_num = 0.0
        return (budget_val_num, str(stats.get('model_type')), str(stats.get('training_regime')))

    sorted_results = sorted(aggregated_results.items(), key=sort_key)
    
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

def print_experiment_console_summary(all_results, start_time, end_time, output_file='outputs/experiment_summary.txt'):
    """Print experiment summary to console and save to file"""
    from pathlib import Path
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    total_time = end_time - start_time
    
    # Build summary content
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("🎯 EXPERIMENT SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append(f"⏱️ Total experiment time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    summary_lines.append("")

    # Find best model by test accuracy
    successful_results = [r for r in all_results if r.get('success', False)]
    failed_results = [r for r in all_results if not r.get('success', False)]
    
    summary_lines.append(f"📊 EXPERIMENT OVERVIEW:")
    summary_lines.append(f"   Total experiments: {len(all_results)}")
    summary_lines.append(f"   Successful: {len(successful_results)}")
    summary_lines.append(f"   Failed: {len(failed_results)}")
    summary_lines.append("")
    
    if successful_results:
        best = max(successful_results, key=lambda x: float(x.get('test_accuracy', 0) or 0))
        summary_lines.append("🏆 BEST PERFORMING MODEL:")
        summary_lines.append(f"   Model Type: {best['model_type'].upper()}")
        summary_lines.append(f"   Training Regime: {best.get('training_regime', 'N/A')}")
        summary_lines.append(f"   Test Accuracy: {best['test_accuracy']:.2f}%")
        summary_lines.append(f"   Model Name: {best.get('model_name', 'N/A')}")
        summary_lines.append(f"   Model Path: {best.get('best_model_path', 'N/A')}")
        
        # Add additional metrics if available
        if best.get('precision_macro') is not None:
            summary_lines.append(f"   Precision (macro): {best.get('precision_macro', 0):.3f}")
        if best.get('recall_macro') is not None:
            summary_lines.append(f"   Recall (macro): {best.get('recall_macro', 0):.3f}")
        if best.get('f1_macro') is not None:
            summary_lines.append(f"   F1-Score (macro): {best.get('f1_macro', 0):.3f}")
        if best.get('matthews_corrcoef') is not None:
            summary_lines.append(f"   Matthews Corr. Coef: {best.get('matthews_corrcoef', 0):.3f}")
    else:
        summary_lines.append("❌ No successful models to report best from.")
    
    summary_lines.append("")
    summary_lines.append("📋 DETAILED EXPERIMENT RESULTS:")
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'#':<3} {'Model':<10} {'Regime':<12} {'Budget':<10} {'Train Acc':<10} {'Test Acc':<10} {'Time (s)':<10} {'Model Name':<30}")
    summary_lines.append("-" * 80)
    
    for idx, r in enumerate(successful_results, 1):
        try:
            train_acc = float(r.get('train_accuracy', 0) or 0)
        except Exception:
            train_acc = 0.0
        try:
            test_acc = float(r.get('test_accuracy', 0) or 0)
        except Exception:
            test_acc = 0.0
        try:
            time_sec = float(r.get('time', 0) or 0)
        except Exception:
            time_sec = 0.0
        budget_mode = r.get('few_shot_info', {}).get('mode', 'N/A')
        budget_value = r.get('few_shot_info', {}).get('value', 'N/A')
        budget_str = f"{budget_mode}={budget_value}"
        summary_lines.append(f"{idx:<3} {r['model_type']:<10} {r.get('training_regime', 'N/A'):<12} {budget_str:<10} {train_acc:<10.2f} {test_acc:<10.2f} {time_sec:<10.1f} {r.get('model_name', ''):<30}")
    
    # Add failed experiments if any
    if failed_results:
        summary_lines.append("")
        summary_lines.append("❌ FAILED EXPERIMENTS:")
        summary_lines.append("-" * 80)
        for idx, r in enumerate(failed_results, 1):
            summary_lines.append(f"{idx:<3} {r['model_type']:<10} {r.get('training_regime', 'N/A'):<12} Error: {r.get('error', 'Unknown error')[:50]}")
    
    # Add detailed comparison results section
    summary_lines.append("")
    summary_lines.append("="*80)
    summary_lines.append("📊 DETAILED COMPARISON RESULTS")
    summary_lines.append("="*80)
    
    # Group results by budget for better organization
    budget_groups = {}
    for result in successful_results:
        budget = result.get('budget_value', 0) * 100
        if budget not in budget_groups:
            budget_groups[budget] = []
        budget_groups[budget].append(result)
    
    for budget_pct in sorted(budget_groups.keys()):
        budget_results = budget_groups[budget_pct]
        summary_lines.append(f"\n🎯 LABEL BUDGET: {int(budget_pct)}%")
        summary_lines.append("-" * 60)
        
        for result in budget_results:
            model_type = result['model_type'].upper()
            regime = result.get('training_regime', 'N/A')
            model_name = result.get('model_name', 'N/A')
            
            train_acc = result.get('train_accuracy', 0)
            test_acc = result.get('test_accuracy', 0)
            best_val = result.get('best_val_acc', 0)
            time_sec = result.get('time', 0)
            
            # Get sample information
            train_samples = result.get('train_samples', 0)
            labeled_samples = result.get('labeled_samples', 0)
            val_samples = result.get('val_samples', 0)
            test_samples = result.get('test_samples', 0)
            
            # Calculate label ratio
            label_ratio = f"{labeled_samples}/{train_samples} ({labeled_samples/train_samples*100:.1f}%)" if train_samples > 0 else "N/A"
            
            # Get config info
            batch_size = result.get('batch_size', 'N/A')
            epochs = result.get('epochs', 'N/A')
            image_size = result.get('image_size', 'N/A')
            if isinstance(image_size, tuple):
                image_size = f"({image_size[0]}, {image_size[1]})"
            
            summary_lines.append(f" {model_type}: {model_name}")
            summary_lines.append(f"     Train Accuracy: {train_acc:.2f}%")
            summary_lines.append(f"     Test Accuracy: {test_acc:.2f}%")
            summary_lines.append(f"     Best Val: {best_val:.2f}%")
            summary_lines.append(f"     Time: {time_sec:.1f}s")
            summary_lines.append(f"     Samples: {train_samples} total train ({labeled_samples} labeled), {val_samples} val, {test_samples} test")
            summary_lines.append(f"     Label ratio: {label_ratio}")
            summary_lines.append(f"     Config: batch_size={batch_size}, image_size={image_size}, epochs={epochs}")
            
            # Add few-shot info
            few_shot_mode = result.get('few_shot_mode', 'N/A')
            few_shot_value = result.get('few_shot_value', 'N/A')
            summary_lines.append(f"    Few-shot: {few_shot_mode} ({few_shot_value})")
            summary_lines.append("")
    
    # Add winner analysis
    if successful_results:
        summary_lines.append("🏆 WINNER ANALYSIS:")
        summary_lines.append("-" * 40)
        
        # Find best model overall
        best_overall = max(successful_results, key=lambda x: float(x.get('test_accuracy', 0) or 0))
        best_acc = best_overall.get('test_accuracy', 0)
        
        # Find second best
        sorted_results = sorted(successful_results, key=lambda x: float(x.get('test_accuracy', 0) or 0), reverse=True)
        if len(sorted_results) > 1:
            second_best = sorted_results[1]
            second_acc = second_best.get('test_accuracy', 0)
            gap = best_acc - second_acc
            
            summary_lines.append(f"WINNER: {best_overall['model_type'].upper()} ({best_acc:.2f}%)")
            summary_lines.append(f"Performance gap: {gap:.2f}% advantage")
        else:
            summary_lines.append(f"WINNER: {best_overall['model_type'].upper()} ({best_acc:.2f}%)")
        
        summary_lines.append("")
    
    summary_lines.append("="*80)
    summary_lines.append("📁 OUTPUT FILES:")
    summary_lines.append("   - Detailed results: outputs/comparison_results/")
    summary_lines.append("   - Model checkpoints: outputs/checkpoints/")
    summary_lines.append("   - Training logs: outputs/logs/")
    summary_lines.append("   - Plots and visualizations: outputs/plots/")
    summary_lines.append("="*80)
    
    # Join all lines
    summary_text = "\n".join(summary_lines)
    
    # Print to console
    print(summary_text)
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"\n📄 Experiment summary saved to: {output_file}")
    except Exception as e:
        print(f"⚠️ Could not save experiment summary to file: {e}")
    
    return output_file

def main():
    """Main comparison function with MULTIPLE SEEDS and TRAINING REGIMES"""
    print("🚀 QUICK TEST MODE: 1000 samples, reduced epochs (~1-2 hours)")
    print("🔧 For full experiment: change SAMPLE_SIZE=10000, EPOCHS back to (15,25,25)")
    
    # ⏱️ Track experiment time
    start_time = time.time()
    
    # Initialize comparison logger
    comparison_logger = ComparisonLogger()
    
    # 📊 AUTO-DETECT DATASET SIZE
    temp_train, temp_labels, temp_val, temp_val_labels, temp_test, temp_test_labels = data_splitter.split_clean_dataset(
        pickle_path=config_paths.CLEAN_DATASET_PICKLE,
        base_data_dir=config_paths.BASE_DATA_DIR,
        few_shot_mode=None,  # Just for size detection, actual few-shot applied later
        data_logger=DataSplitterLogger("dataset_overview")
    )
    total_dataset_size = len(temp_train) + len(temp_val) + len(temp_test)
    
    # 📊 DATASET SIZE Configuration:
    # SAMPLE_SIZE = 10000                            # Full experiment (12-15 hours with old batch 64)
    # SAMPLE_SIZE = 4000                             # ✅ OPTIMIZED: ~6-7 hours with batch 256 + grad accum
    # SAMPLE_SIZE = 1000                             # Quick test (~1.5 hours)
    SAMPLE_SIZE = 500                             # Balanced: good results in <8 hours
    
    # 🎯 CRITICAL: Label efficiency analysis - how much labeled data is needed?
    LABEL_BUDGETS = [
        # ('percentage', 0.05),  # 5% of labels visible (very low-label)
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
        'vit': ['supervised']   # ViT: supervised only (training from scratch)
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
    
    # Statistical summary removed (only 1 seed, no variance to report)
    # print_statistical_summary(aggregated_results)  # Uncomment if using 3+ seeds
    
    end_time = time.time()
    print_experiment_console_summary(all_results, start_time, end_time)

if __name__ == "__main__":
    main() 