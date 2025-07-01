#!/usr/bin/env python3
"""
Compare CNN vs DINOv2 on GPU with 100 samples
Uses respective configs: config.py for CNN, config_dinov2.py for DINOv2
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import time
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# Add thesis_codes to path
sys.path.append(os.path.dirname(__file__))

# Import all configs
import config as config_cnn
import config_dinov2
import training
from model_setup import create_model

# Force GPU usage if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def get_config_for_model(model_name):
    """Get the appropriate config for the model"""
    if 'dinov2' in model_name.lower():
        return config_dinov2
    else:
        return config_cnn

def create_synthetic_dataset(num_samples=100, num_classes=3, image_size=(224, 224)):
    """Create synthetic dataset for testing"""
    print(f"🎯 Creating synthetic dataset:")
    print(f"   Samples: {num_samples}")
    print(f"   Classes: {num_classes}")
    print(f"   Image size: {image_size}")
    
    # Create balanced dataset
    samples_per_class = num_samples // num_classes
    remainder = num_samples % num_classes
    
    X_list = []
    y_list = []
    
    for class_idx in range(num_classes):
        class_samples = samples_per_class + (1 if class_idx < remainder else 0)
        
        # Create class-specific patterns to make learning possible
        if class_idx == 0:
            # Class 0: Mostly red-ish images
            class_X = torch.randn(class_samples, 3, *image_size) * 0.5
            class_X[:, 0, :, :] += 1.0  # Red channel boost
        elif class_idx == 1:
            # Class 1: Mostly green-ish images
            class_X = torch.randn(class_samples, 3, *image_size) * 0.5
            class_X[:, 1, :, :] += 1.0  # Green channel boost
        else:
            # Class 2: Mostly blue-ish images
            class_X = torch.randn(class_samples, 3, *image_size) * 0.5
            class_X[:, 2, :, :] += 1.0  # Blue channel boost
        
        class_y = torch.full((class_samples,), class_idx, dtype=torch.long)
        
        X_list.append(class_X)
        y_list.append(class_y)
        
        print(f"   Class {class_idx}: {class_samples} samples")
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    # Shuffle the dataset
    indices = torch.randperm(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

def create_data_loaders(X, y, config):
    """Create train/val data loaders"""
    # Split into train/val (80/20 split for small dataset)
    train_size = int(0.8 * len(X))
    val_size = len(X) - train_size
    
    train_X, val_X = X[:train_size], X[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]
    
    print(f"📊 Data splits:")
    print(f"   Training: {len(train_X)} samples")
    print(f"   Validation: {len(val_X)} samples")
    
    # Create datasets and loaders
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    return train_loader, val_loader

def setup_optimizer_and_scheduler(model, config):
    """Setup optimizer and scheduler based on config"""
    # Create optimizer
    if config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            **config.OPTIMIZER_PARAMS['sgd']
        )
    elif config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            **config.OPTIMIZER_PARAMS['adam']
        )
    else:  # adamw
        optimizer = optim.AdamW(
            model.parameters(), 
            **config.OPTIMIZER_PARAMS['adamw']
        )
    
    # Create scheduler
    if config.SCHEDULER == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            **config.SCHEDULER_PARAMS['plateau']
        )
    elif config.SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            **config.SCHEDULER_PARAMS['cosine']
        )
    elif config.SCHEDULER == 'cosinerestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            **config.SCHEDULER_PARAMS['cosinerestarts']
        )
    elif config.SCHEDULER == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            **config.SCHEDULER_PARAMS['linear']
        )
    else:  # step
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            **config.SCHEDULER_PARAMS['step']
        )
    
    return optimizer, scheduler

def test_model(model_name, train_loader, val_loader, test_loader, config, class_names=None, class_to_idx=None, num_classes=3, train_size=60, val_size=20, test_size=20, total_samples_used=100):
    """Test a specific model"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING {model_name.upper()}")
    print(f"{'='*60}")
    print(f"📋 Using config: {'config_dinov2.py' if 'dinov2' in model_name.lower() else 'config.py'}")
    
    start_time = time.time()
    
    try:
        # Apply model-specific configurations if available (especially for DINOv2)
        if hasattr(config, 'MODEL_CONFIGS') and model_name in config.MODEL_CONFIGS:
            model_specific_config = config.MODEL_CONFIGS[model_name]
            print(f"🎯 Applying model-specific config for {model_name}:")
            
            # Override config settings with model-specific ones
            original_lr = config.LEARNING_RATE
            original_batch = config.BATCH_SIZE
            original_wd = config.WEIGHT_DECAY
            original_dropout = config.DROPOUT
            
            config.LEARNING_RATE = model_specific_config.get('learning_rate', config.LEARNING_RATE)
            config.BATCH_SIZE = model_specific_config.get('batch_size', config.BATCH_SIZE)
            config.WEIGHT_DECAY = model_specific_config.get('weight_decay', config.WEIGHT_DECAY)
            config.DROPOUT = model_specific_config.get('dropout', config.DROPOUT)
            
            print(f"   Learning Rate: {original_lr} → {config.LEARNING_RATE}")
            print(f"   Batch Size: {original_batch} → {config.BATCH_SIZE}")
            print(f"   Weight Decay: {original_wd} → {config.WEIGHT_DECAY}")
            print(f"   Dropout: {original_dropout} → {config.DROPOUT}")
            
            # Update optimizer params with new learning rate
            for opt_name in config.OPTIMIZER_PARAMS:
                config.OPTIMIZER_PARAMS[opt_name]['lr'] = config.LEARNING_RATE
                config.OPTIMIZER_PARAMS[opt_name]['weight_decay'] = config.WEIGHT_DECAY

        # Create model using the appropriate config
        print(f"🔧 Creating model {model_name}...")
        model = create_model(
            num_classes=num_classes, 
            model_name=model_name, 
            config=config
        )
        
        # Verify model was created properly
        if not hasattr(model, 'parameters'):
            raise ValueError(f"Model creation failed - returned object has no 'parameters' method: {type(model)}")
        
        # Move model to device
        model = model.to(device)
        
        print(f"⚙️  Model Configuration:")
        print(f"   Model: {model_name}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Setup optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = setup_optimizer_and_scheduler(model, config)
        
        # 🚀 Apply advanced features from config
        compile_enabled = hasattr(config, 'COMPILE_MODEL') and config.COMPILE_MODEL and hasattr(torch, 'compile')
        if compile_enabled:
            compile_mode = getattr(config, 'COMPILE_MODE', 'default')
            
            # Use safer compilation mode for DINOv2 models
            if 'dinov2' in model_name.lower():
                # DINOv2 models have complex architectures that can cause device issues with torch.compile
                print(f"⚠️  DINOv2 model detected: Disabling torch.compile to avoid device placement issues")
                compile_enabled = False
            else:
                print(f"⚡ Compiling model with mode: {compile_mode}")
                
                try:
                    compiled_model = torch.compile(model, mode=compile_mode)
                    
                    # Verify the compiled model still has necessary PyTorch methods
                    if hasattr(compiled_model, 'train') and hasattr(compiled_model, 'eval') and hasattr(compiled_model, 'parameters'):
                        model = compiled_model
                        print(f"   ✅ Model compilation successful")
                    else:
                        print(f"   ⚠️  Compiled model missing PyTorch methods, using uncompiled model")
                        print(f"   Available methods: {[attr for attr in dir(compiled_model) if not attr.startswith('_')][:10]}...")
                        compile_enabled = False
                except Exception as e:
                    print(f"   ❌ Model compilation failed: {e}")
                    compile_enabled = False
        
        # Mixed precision setup
        use_amp = getattr(config, 'MIXED_PRECISION', False) and device.type == 'cuda'
        if use_amp:
            print(f"⚡ Mixed precision training enabled")
        
        # Gradient checkpointing for DINOv2
        if hasattr(config, 'USE_GRADIENT_CHECKPOINTING') and config.USE_GRADIENT_CHECKPOINTING:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                print(f"💾 Gradient checkpointing enabled")
        
        print(f"⚙️  Model Configuration Summary:")
        print(f"   Model: {model_name}")
        print(f"   Batch Size: {config.BATCH_SIZE}")
        print(f"   Image Size: {config.IMAGE_SIZE}")
        print(f"   Learning Rate: {config.LEARNING_RATE}")
        print(f"   Weight Decay: {config.WEIGHT_DECAY}")
        print(f"   Optimizer: {config.OPTIMIZER}")
        print(f"   Scheduler: {config.SCHEDULER}")
        print(f"   Epochs: {config.EPOCHS}")
        print(f"   Early Stopping: {getattr(config, 'EARLY_STOPPING_PATIENCE', 'N/A')} patience")
        print(f"   Few-shot: {getattr(config, 'FEW_SHOT_MODE', 'Disabled')}")
        print(f"   Mixed Precision: {use_amp}")
        print(f"   Model Compilation: {compile_enabled}")
        
        # Create class mappings (use provided ones or default for synthetic data)
        if class_names is None:
            class_names = [f'class_{i}' for i in range(num_classes)]
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        print(f"🚀 Starting training...")
        
        # Note: use_amp already calculated above from config settings
        result = training.train_model(
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
            use_amp=use_amp,
            config_module=config
        )
        
        # 🧪 IMPORTANT: Test evaluation on test set
        print(f"📊 Evaluating on test set...")
        from evaluation import comprehensive_test_evaluation
        
        test_eval_result = comprehensive_test_evaluation(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            model_name=model_name,
            config_module=config,
            use_amp=use_amp
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"✅ {model_name} training completed!")
        print(f"⏱️  Training time: {training_time:.1f} seconds")
        print(f"📊 Training Results:")
        for key, value in result.items():
            if key not in ['log_file', 'results_file']:  # Skip file paths
                print(f"   {key}: {value}")
        
        print(f"🎯 Test Evaluation Results:")
        print(f"   Test Accuracy: {test_eval_result['test_accuracy']:.2f}%")
        print(f"   Test Samples: {len(test_eval_result['y_true'])}")
        if 'per_class_accuracy' in test_eval_result:
            print(f"   Per-class Accuracy:")
            for class_name, class_info in test_eval_result['per_class_accuracy'].items():
                print(f"     {class_name}: {class_info['accuracy']:.2f}% ({class_info['sample_count']} samples)")
        
        # Verify model device placement for complex architectures like DINOv2
        if 'dinov2' in model_name.lower():
            # Ensure all model parameters are on the correct device
            device_params = set()
            for name, param in model.named_parameters():
                device_params.add(param.device)
            
            device_buffers = set()
            for name, buffer in model.named_buffers():
                device_buffers.add(buffer.device)
            
            print(f"🔧 Model device verification:")
            print(f"   Parameter devices: {device_params}")
            print(f"   Buffer devices: {device_buffers}")
            
            if len(device_params) > 1 or len(device_buffers) > 1:
                print(f"   ⚠️  Mixed device placement detected, ensuring all components are on {device}")
                model = model.to(device)
        
        print("✅ Model setup completed successfully.")
        
        return {
            'model_name': model_name,
            'config_used': 'config_dinov2.py' if 'dinov2' in model_name.lower() else 'config.py',
            'training_time': float(training_time),
            'success': True,
            'best_val_accuracy': result.get('best_val_acc', 0),
            'final_train_accuracy': result.get('train_accuracies', [0])[-1] if result.get('train_accuracies') else 0,
            'final_val_accuracy': result.get('val_accuracies', [0])[-1] if result.get('val_accuracies') else 0,
            'test_accuracy': test_eval_result['test_accuracy'],  # 🎯 IMPORTANT: Test accuracy from evaluation.py
            'test_samples': len(test_eval_result['y_true']),
            'per_class_test_accuracy': test_eval_result.get('per_class_accuracy', {}),
            # 🎛️ Config tracking for summary
            'epochs_used': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'optimizer': config.OPTIMIZER,
            'scheduler': config.SCHEDULER,
            'few_shot_mode': getattr(config, 'FEW_SHOT_MODE', 'Disabled'),
            'few_shot_value': getattr(config, 'FEW_SHOT_VALUE', 'N/A'),
            'early_stopping_used': getattr(config, 'EARLY_STOPPING_PATIENCE', 'N/A'),
            'mixed_precision_used': use_amp,
            'model_compilation_used': compile_enabled,
            'gradient_checkpointing': hasattr(config, 'USE_GRADIENT_CHECKPOINTING') and config.USE_GRADIENT_CHECKPOINTING,
            'image_size': config.IMAGE_SIZE,
            'weight_decay': config.WEIGHT_DECAY,
            # 📊 Data size tracking
            'train_size_used': train_size,
            'val_size_used': val_size,
            'test_size_used': test_size,
            'total_samples_used': total_samples_used,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples_split': len(test_loader.dataset),
            'test_evaluation_log': test_eval_result.get('log_file', ''),
            'test_evaluation_results': test_eval_result.get('results_file', ''),
            **result
        }
        
    except Exception as e:
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"❌ {model_name} training failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'model_name': model_name,
            'config_used': 'config_dinov2.py' if 'dinov2' in model_name.lower() else 'config.py',
            'training_time': float(training_time),
            'success': False,
            'error': str(e)
        }

def calculate_split_sizes(total_samples, train_mode, train_value, val_mode, val_value, test_mode, test_value, num_classes):
    """Calculate train/val/test split sizes based on mode and value"""
    
    # Calculate each split size
    if train_mode == 'percentage':
        train_size = int(total_samples * train_value)
    else:  # fixed
        train_size = train_value
    
    if val_mode == 'percentage':
        val_size = int(total_samples * val_value)  
    else:  # fixed
        val_size = val_value
        
    if test_mode == 'percentage':
        test_size = int(total_samples * test_value)
    else:  # fixed
        test_size = test_value
    
    # Ensure minimum samples per split (at least num_classes for proper stratification)
    train_size = max(train_size, num_classes)
    val_size = max(val_size, num_classes)
    test_size = max(test_size, num_classes)
    
    # Ensure we don't exceed total samples
    total_requested = train_size + val_size + test_size
    if total_requested > total_samples:
        print(f"⚠️  Warning: Requested {total_requested} samples but only {total_samples} available")
        print(f"   Scaling down proportionally...")
        
        scale_factor = total_samples / total_requested
        train_size = max(int(train_size * scale_factor), num_classes)
        val_size = max(int(val_size * scale_factor), 1)
        test_size = max(int(test_size * scale_factor), 1)
        
        # Final adjustment to exactly match total_samples
        current_total = train_size + val_size + test_size
        if current_total < total_samples:
            # Add remaining to train set
            train_size += (total_samples - current_total)
        elif current_total > total_samples:
            # Remove from largest set
            if train_size >= val_size and train_size >= test_size:
                train_size -= (current_total - total_samples)
            elif val_size >= test_size:
                val_size -= (current_total - total_samples)
            else:
                test_size -= (current_total - total_samples)
    
    return train_size, val_size, test_size

def safe_format_number(value, decimal_places=1):
    """Safely format a number, handling strings and None values"""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimal_places}f}"
    except (ValueError, TypeError):
        return str(value)

def main():
    """Main comparison function"""
    print("="*80)
    print("🥊 CNN vs DINOv2 COMPARISON - FLEXIBLE DATA SIZE + TRUE FEW-SHOT LEARNING ON GPU")
    print("="*80)
    
    # 🎛️ QUICK CONFIG OVERRIDES (uncomment to modify for testing)
    config_cnn.EPOCHS = 5           # Override epochs for quicker testing
    config_dinov2.EPOCHS = 5        # Override epochs for quicker testing
    #config_cnn.FEW_SHOT_MODE = 'percentage'  # Enable few-shot
    #config_cnn.FEW_SHOT_VALUE = 0.4          # Use 10% of data
    #config_dinov2.FEW_SHOT_MODE = 'percentage' # Different few-shot mode
    #config_dinov2.FEW_SHOT_VALUE = 0.4         # 20 samples per class
    
    # 📊 INDEPENDENT DATA SIZE CONTROL FOR EACH SPLIT
    # You can now control train/val/test sizes independently!
    
    # TRAINING SET SIZE
    TRAIN_SIZE_MODE = 'fixed'         # Options: 'percentage', 'fixed'
    TRAIN_SIZE_VALUE = 1000            # For 'percentage': 0.1 = 10%, for 'fixed': number of samples
    
    # VALIDATION SET SIZE
    VAL_SIZE_MODE = 'percentage'           # Options: 'percentage', 'fixed'
    VAL_SIZE_VALUE = 0.15               # For 'percentage': 0.05 = 5%, for 'fixed': number of samples
    
    # TEST SET SIZE
    TEST_SIZE_MODE = 'percentage'          # Options: 'percentage', 'fixed'
    TEST_SIZE_VALUE = 0.15              # For 'percentage': 0.15 = 15%, for 'fixed': number of samples
    
    # Alternative configurations you can try:
    # TRAIN_SIZE_MODE = 'percentage'; TRAIN_SIZE_VALUE = 0.6  # 60% of total data
    # VAL_SIZE_MODE = 'percentage'; VAL_SIZE_VALUE = 0.2      # 20% of total data  
    # TEST_SIZE_MODE = 'percentage'; TEST_SIZE_VALUE = 0.2    # 20% of total data
    
    # 🎯 ENABLE FEW-SHOT LEARNING (uncomment to activate):
    # config_cnn.FEW_SHOT_MODE = 'percentage'      # Enable few-shot for CNN
    # config_cnn.FEW_SHOT_VALUE = 0.1             # Only 10% of data has labels
    # config_dinov2.FEW_SHOT_MODE = 'percentage'   # Enable few-shot for DINOv2  
    # config_dinov2.FEW_SHOT_VALUE = 0.1          # Only 10% of data has labels
    
    # 🎯 EXAMPLE: 200 samples + 10% few-shot
    # Result: Model sees 200 images, but only 20 have labels for training!
    
    print(f"📋 Configuration Summary:")
    print(f"   CNN Epochs: {config_cnn.EPOCHS}")
    print(f"   DINOv2 Epochs: {config_dinov2.EPOCHS}")
    print(f"   CNN Few-shot: {getattr(config_cnn, 'FEW_SHOT_MODE', 'Disabled')}")
    print(f"   DINOv2 Few-shot: {getattr(config_dinov2, 'FEW_SHOT_MODE', 'Disabled')}")
    print(f"   Data Split Configuration:")
    print(f"     Training: {TRAIN_SIZE_MODE} - {TRAIN_SIZE_VALUE}")
    print(f"     Validation: {VAL_SIZE_MODE} - {VAL_SIZE_VALUE}")
    print(f"     Test: {TEST_SIZE_MODE} - {TEST_SIZE_VALUE}")
    
    if TRAIN_SIZE_MODE == 'percentage':
        print(f"   Expected train samples: ~{int(TRAIN_SIZE_VALUE * 100)}% of dataset")
    else:
        print(f"   Expected train samples: {TRAIN_SIZE_VALUE}")
    
    if VAL_SIZE_MODE == 'percentage':
        print(f"   Expected val samples: ~{int(VAL_SIZE_VALUE * 100)}% of dataset")
    else:
        print(f"   Expected val samples: {VAL_SIZE_VALUE}")
        
    if TEST_SIZE_MODE == 'percentage':
        print(f"   Expected test samples: ~{int(TEST_SIZE_VALUE * 100)}% of dataset")
    else:
        print(f"   Expected test samples: {TEST_SIZE_VALUE}")
    
    # Models to test with their respective configs (you can change these later)
    models_to_test = [
        ('efficientnet_b3', config_cnn),   # CNN with config.py
        ('dinov2_vits14', config_dinov2),  # DINOv2 with config_dinov2.py
    ]
    
    # Other available models you can use:
    # ('efficientnet_b0', config_cnn),   # Faster CNN
    # ('dinov2_vitb14', config_dinov2),  # Base DINOv2
    # ('dinov2_vitl14', config_dinov2),  # Large DINOv2
    # ('dinov2_vitg14', config_dinov2),  # Giant DINOv2
    
    results = []
    
    for model_name, model_config in models_to_test:
        print(f"\n🔧 Preparing {model_name} with {model_config.__name__}")
        
        # 📁 REAL DATA LOADING FROM CLEANED PICKLE FILE:
        from data_splitter import split_clean_dataset
        from dataloader_setup import create_dataloaders
        
        # Load data from cleaned pickle file (avoids corrupted files)
        data_dir = '/teamspace/studios/this_studio/crop_pest_data'  # Base data directory
        pickle_path = '/teamspace/studios/this_studio/thesis_codes/clean_dataset.pkl'  # Cleaned dataset pickle
        
        print(f"📁 Loading cleaned dataset from pickle file: {pickle_path}")
        
        # Use data_splitter to load the cleaned dataset (this avoids corrupted files)
        # We'll get the full dataset first, then apply our custom splitting
        temp_train_paths, temp_train_labels, temp_val_paths, temp_val_labels, temp_test_paths, temp_test_labels = split_clean_dataset(
            pickle_path=pickle_path,
            test_size=0.15,  # Temporary split just to get all data
            val_size=0.15,   # We'll re-split later with our custom logic
            random_state=model_config.RANDOM_STATE,
            base_data_dir=data_dir,
            few_shot_mode=None,  # We'll handle few-shot separately
            few_shot_value=0.1
        )
        
        # Combine all splits back into full dataset (since we want to apply custom splitting)
        image_paths = temp_train_paths + temp_val_paths + temp_test_paths
        labels = temp_train_labels + temp_val_labels + temp_test_labels
        
        # Create class mappings
        class_names = sorted(list(set(labels)))
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        label_indices = [class_to_idx[label] for label in labels]
        num_classes = len(class_names)
        
        print(f"📊 Dataset info (from cleaned pickle):")
        print(f"   Classes found: {num_classes}")
        print(f"   Class names: {class_names}")
        print(f"   Total samples: {len(image_paths)} (corrupted files already removed)")
        print(f"   Label range: [{min(label_indices)}, {max(label_indices)}]")
        
        # 📊 Calculate independent split sizes
        original_total_samples = len(image_paths)
        
        # Use the new independent split configuration
        train_size, val_size, test_size = calculate_split_sizes(
            total_samples=original_total_samples,
            train_mode=TRAIN_SIZE_MODE, train_value=TRAIN_SIZE_VALUE,
            val_mode=VAL_SIZE_MODE, val_value=VAL_SIZE_VALUE,
            test_mode=TEST_SIZE_MODE, test_value=TEST_SIZE_VALUE,
            num_classes=num_classes
        )
        
        print(f"📊 Independent Split Configuration:")
        print(f"   Total available samples: {original_total_samples}")
        print(f"   Planned splits:")
        print(f"     Training: {train_size} samples")
        print(f"     Validation: {val_size} samples") 
        print(f"     Test: {test_size} samples")
        print(f"     Total requested: {train_size + val_size + test_size} samples")
        
        # Calculate total samples needed
        total_needed = train_size + val_size + test_size
        
        # ⚠️ Check if we have enough data
        if total_needed > original_total_samples:
            print(f"   ❌ ERROR: Requested {total_needed} samples but only {original_total_samples} available!")
            print(f"   💡 Solution: Reduce split sizes or switch to percentage mode")
            print(f"   💡 Example: TRAIN_SIZE_MODE='percentage', TRAIN_SIZE_VALUE=0.7")
            continue  # Skip this model and continue with next
        
        # Sample data if we need fewer samples than available
        if total_needed < len(image_paths):
            random.seed(model_config.RANDOM_STATE)
            combined_data = list(zip(image_paths, label_indices))
            sampled_data = random.sample(combined_data, total_needed)
            image_paths, label_indices = zip(*sampled_data)
            image_paths, label_indices = list(image_paths), list(label_indices)
            print(f"   Sampled from {original_total_samples} → {len(image_paths)} samples")
        
        # 🎯 Apply few-shot learning (hide labels from most data)
        original_label_indices = label_indices.copy()  # Keep original labels for test evaluation
        labeled_samples = len(image_paths)  # Default: all samples have labels
        
        # Check if few-shot is enabled in the config
        few_shot_enabled = (hasattr(model_config, 'FEW_SHOT_MODE') and 
                           model_config.FEW_SHOT_MODE is not None and 
                           model_config.FEW_SHOT_MODE != False)
        
        print(f"🔍 Few-shot check:")
        print(f"   Has FEW_SHOT_MODE: {hasattr(model_config, 'FEW_SHOT_MODE')}")
        if hasattr(model_config, 'FEW_SHOT_MODE'):
            print(f"   FEW_SHOT_MODE value: {getattr(model_config, 'FEW_SHOT_MODE', None)}")
            print(f"   FEW_SHOT_VALUE: {getattr(model_config, 'FEW_SHOT_VALUE', None)}")
        
        if few_shot_enabled:
            print(f"🎯 Few-shot learning ENABLED:")
            print(f"   Mode: {model_config.FEW_SHOT_MODE}")
            print(f"   Value: {model_config.FEW_SHOT_VALUE}")
            
            # Handle different FEW_SHOT_MODE values
            if model_config.FEW_SHOT_MODE == 'percentage' or model_config.FEW_SHOT_MODE == True:
                labeled_samples = int(len(image_paths) * model_config.FEW_SHOT_VALUE)
                print(f"   Treating as 'percentage' mode: {model_config.FEW_SHOT_VALUE*100:.1f}% of {len(image_paths)} samples")
            elif model_config.FEW_SHOT_MODE == 'per_class':
                labeled_samples = min(int(num_classes * model_config.FEW_SHOT_VALUE), len(image_paths))
                print(f"   Using 'per_class' mode: {model_config.FEW_SHOT_VALUE} samples per class")
            else:
                # Fallback to percentage mode
                labeled_samples = int(len(image_paths) * model_config.FEW_SHOT_VALUE)
                print(f"   Unknown mode '{model_config.FEW_SHOT_MODE}', defaulting to percentage")
            
            labeled_samples = max(labeled_samples, num_classes)  # At least 1 per class
            labeled_samples = min(labeled_samples, len(image_paths))  # Don't exceed total
            
            print(f"   📊 Data with labels: {labeled_samples}/{len(image_paths)} ({labeled_samples/len(image_paths)*100:.1f}%)")
            print(f"   📊 Hidden labels: {len(image_paths) - labeled_samples} samples")
            
            # Create mask for which samples have labels
            random.seed(model_config.RANDOM_STATE)
            labeled_indices = random.sample(range(len(image_paths)), labeled_samples)
            labeled_mask = [i in labeled_indices for i in range(len(image_paths))]
            
            # Apply few-shot masking - hide labels for unlabeled samples
            few_shot_labels = []
            for i, (original_label, has_label) in enumerate(zip(label_indices, labeled_mask)):
                if has_label:
                    few_shot_labels.append(original_label)  # Keep original label
                else:
                    few_shot_labels.append(-1)  # Hide label (use -1 as unlabeled marker)
            
            label_indices = few_shot_labels
            print(f"   ✅ Few-shot masking applied: {labeled_samples} labeled, {len(image_paths) - labeled_samples} unlabeled")
        else:
            print(f"🎯 Few-shot learning: DISABLED (all {len(image_paths)} samples have labels)")
            print(f"   To enable few-shot, uncomment the config lines in main()")
        
        # 📊 Create independent train/val/test splits based on calculated sizes
        print(f"📊 Creating independent data splits:")
        
        # Shuffle all data first
        random.seed(model_config.RANDOM_STATE)
        combined_data = list(zip(image_paths, label_indices, original_label_indices))
        random.shuffle(combined_data)
        
        # Split according to calculated sizes
        train_end = train_size
        val_end = train_size + val_size
        test_end = train_size + val_size + test_size
        
        # Create splits
        train_data = combined_data[:train_end]
        val_data = combined_data[train_end:val_end]
        test_data = combined_data[val_end:test_end]
        
        # Extract paths and labels for each split
        if train_data:
            train_paths, train_labels_masked, train_labels_original = zip(*train_data)
            train_paths, train_labels_masked = list(train_paths), list(train_labels_masked)
        else:
            train_paths, train_labels_masked = [], []
            
        if val_data:
            val_paths, val_labels_masked, val_labels_original = zip(*val_data)
            val_paths, val_labels_masked = list(val_paths), list(val_labels_masked)
        else:
            val_paths, val_labels_masked = [], []
            
        if test_data:
            test_paths, test_labels_masked, test_labels_original = zip(*test_data)
            test_paths, test_labels_original = list(test_paths), list(test_labels_original)
        else:
            test_paths, test_labels_original = [], []
        
        # For few-shot learning: only use labeled samples for training/validation
        if few_shot_enabled:
            print(f"🎯 Applying few-shot masking to train/val splits:")
            # Filter out unlabeled samples (-1) from train/val sets
            train_labeled_indices = [i for i, label in enumerate(train_labels_masked) if label != -1]
            val_labeled_indices = [i for i, label in enumerate(val_labels_masked) if label != -1]
            
            train_paths = [train_paths[i] for i in train_labeled_indices]
            train_labels_masked = [train_labels_masked[i] for i in train_labeled_indices]
            
            val_paths = [val_paths[i] for i in val_labeled_indices]
            val_labels_masked = [val_labels_masked[i] for i in val_labeled_indices]
            
            print(f"   Train samples with labels: {len(train_paths)}/{train_size}")
            print(f"   Val samples with labels: {len(val_paths)}/{val_size}")
            
            # ⚠️ CRITICAL: Ensure we have at least some samples for training and validation
            if len(train_paths) == 0:
                print(f"   ❌ ERROR: No labeled training samples after few-shot filtering!")
                print(f"   💡 Solution: Increase FEW_SHOT_VALUE or disable few-shot learning")
                return None
                
            if len(val_paths) == 0:
                print(f"   ⚠️  WARNING: No labeled validation samples after few-shot filtering!")
                print(f"   💡 Using 1 training sample for validation to avoid division by zero")
                # Use one training sample for validation to prevent empty validation set
                val_paths = [train_paths[0]]
                val_labels_masked = [train_labels_masked[0]]
        
        # Use original labels for train/val (convert from masked labels)
        train_labels = train_labels_masked if not few_shot_enabled else train_labels_masked
        val_labels = val_labels_masked if not few_shot_enabled else val_labels_masked
        test_labels = test_labels_original  # Always use original labels for test
        
        # ✅ FINAL VALIDATION: Ensure no empty datasets
        print(f"📊 Final split validation:")
        print(f"   Training samples: {len(train_paths)}")
        print(f"   Validation samples: {len(val_paths)}")
        print(f"   Test samples: {len(test_paths)}")
        
        if len(train_paths) == 0:
            print(f"   ❌ ERROR: Empty training set!")
            return None
        if len(val_paths) == 0:
            print(f"   ❌ ERROR: Empty validation set!")
            return None
        if len(test_paths) == 0:
            print(f"   ❌ ERROR: Empty test set!")
            return None
        
        # 🔧 FIX: Convert integer labels back to string labels for CustomCropDataset
        # CustomCropDataset expects string class names, not integer indices
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        
        # Convert train/val/test labels from integers back to string class names
        train_labels_str = [idx_to_class[label] for label in train_labels] if train_labels else []
        val_labels_str = [idx_to_class[label] for label in val_labels] if val_labels else []
        test_labels_str = [idx_to_class[label] for label in test_labels]
        
        print(f"📊 Final data splits:")
        print(f"   Training: {len(train_paths)} samples (ONLY labeled data for few-shot)")
        print(f"   Validation: {len(val_paths)} samples (from labeled data)")
        print(f"   Test: {len(test_paths)} samples (ALL data with true labels)")
        print(f"   Total samples used: {len(image_paths)}")
        if hasattr(model_config, 'FEW_SHOT_MODE') and model_config.FEW_SHOT_MODE:
            print(f"   Few-shot effect: Model trains on {len(train_paths)}/{len(image_paths)} samples ({len(train_paths)/len(image_paths)*100:.1f}%)")
        
        print(f"🔧 Label format conversion:")
        print(f"   Train labels: {train_labels[:3] if train_labels else 'None'} → {train_labels_str[:3] if train_labels_str else 'None'}")
        print(f"   Val labels: {val_labels[:3] if val_labels else 'None'} → {val_labels_str[:3] if val_labels_str else 'None'}")
        print(f"   Test labels: {test_labels[:3]} → {test_labels_str[:3]}")
        
        # Check if data splitting was successful
        if train_paths is None or val_paths is None or test_paths is None:
            print(f"   ❌ Data splitting failed for {model_name}, skipping...")
            continue
        
        # Create dataloaders with string labels (not integer indices)
        try:
            train_loader, val_loader, test_loader, _, _, _ = create_dataloaders(
                train_paths, train_labels_str, val_paths, val_labels_str, test_paths, test_labels_str, 
                config_module=model_config, run_batch_test=False)
        except Exception as e:
            print(f"   ❌ DataLoader creation failed for {model_name}: {e}")
            continue
        
        # Verify dataloaders are not empty
        try:
            # Try to get dataset lengths safely
            train_len = 0
            val_len = 0  
            test_len = 0
            
            try:
                train_len = len(train_loader.dataset)  # type: ignore
            except:
                train_len = len(train_paths)
                
            try:
                val_len = len(val_loader.dataset)  # type: ignore
            except:
                val_len = len(val_paths)
                
            try:
                test_len = len(test_loader.dataset)  # type: ignore
            except:
                test_len = len(test_paths)
            
            if train_len == 0 or val_len == 0 or test_len == 0:
                print(f"   ❌ Empty dataloaders detected for {model_name} (train:{train_len}, val:{val_len}, test:{test_len}), skipping...")
                continue
                
            print(f"   ✅ DataLoaders created successfully (train:{train_len}, val:{val_len}, test:{test_len})")
        except Exception as e:
            print(f"   ⚠️  Could not verify dataloader sizes: {e}, proceeding anyway...")
        
        # Test the model with independent split sizes
        result = test_model(
            model_name, train_loader, val_loader, test_loader, model_config, 
            class_names, class_to_idx, num_classes, 
            train_size, val_size, test_size, total_needed
        )
        
        if result is not None:  # Only append successful results
            results.append(result)
        else:
            print(f"   ❌ Model training failed for {model_name}, skipping...")
        
        # Clear GPU cache between models
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Small break between models
        time.sleep(2)
    
    # Print comparison summary
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    for result in results:
        model_name = result['model_name']
        config_used = result.get('config_used', 'Unknown')
        
        if result['success']:
            print(f"\n🏆 {model_name.upper()} ({config_used}):")
            print(f"   ⏱️  Training Time: {safe_format_number(result['training_time'])}s")
            print(f"   🎯 TEST ACCURACY: {safe_format_number(result.get('test_accuracy'), 2)}% ⭐")
            print(f"   📊 Test Samples: {result.get('test_samples', 'N/A')}")
            print(f"   📈 Final Train Accuracy: {safe_format_number(result.get('final_train_accuracy'), 2)}%")
            print(f"   📉 Final Val Accuracy: {safe_format_number(result.get('final_val_accuracy'), 2)}%")
            print(f"   🏅 Best Val Accuracy: {safe_format_number(result.get('best_val_accuracy'), 2)}%")
            print(f"   ⚙️  Config Used:")
            print(f"      Epochs: {result.get('epochs_used', 'N/A')}")
            print(f"      Batch Size: {result.get('batch_size', 'N/A')}")
            print(f"      Learning Rate: {result.get('learning_rate', 'N/A')}")
            print(f"      Optimizer: {result.get('optimizer', 'N/A')}")
            print(f"      Few-shot: {result.get('few_shot_mode', 'N/A')}")
            print(f"      Early Stopping: {result.get('early_stopping_used', 'N/A')}")
            print(f"      Mixed Precision: {result.get('mixed_precision_used', 'N/A')}")
            print(f"   📊 Data Used:")
            print(f"      Total Samples: {result.get('total_samples_used', 'N/A')}")
            print(f"      Planned Splits: {result.get('train_size_used', 'N/A')}/{result.get('val_size_used', 'N/A')}/{result.get('test_size_used', 'N/A')}")
            print(f"      Actual Train/Val/Test: {result.get('train_samples', 'N/A')}/{result.get('val_samples', 'N/A')}/{result.get('test_samples_split', 'N/A')}")
            if result.get('few_shot_mode', 'Disabled') != 'Disabled':
                train_pct = (result.get('train_samples', 0) / result.get('total_samples_used', 1)) * 100
                print(f"      Few-shot training: {train_pct:.1f}% of total data had labels")
        else:
            print(f"\n💥 {model_name.upper()} ({config_used}): FAILED")
            print(f"   ❌ Error: {result.get('error', 'Unknown')}")
            print(f"   ⏱️  Time before failure: {safe_format_number(result['training_time'])}s")
    
    # Analyze results by model type
    cnn_results = [r for r in results if 'dinov2' not in r['model_name'].lower() and r['success']]
    dinov2_results = [r for r in results if 'dinov2' in r['model_name'].lower() and r['success']]
    
    print(f"\n🏅 ANALYSIS:")
    
    if cnn_results:
        best_cnn = max(cnn_results, key=lambda x: x.get('test_accuracy', 0))
        print(f"   🔥 CNN: {best_cnn['model_name']} (TEST: {safe_format_number(best_cnn.get('test_accuracy'), 2)}%)")

    if dinov2_results:
        best_dinov2 = max(dinov2_results, key=lambda x: x.get('test_accuracy', 0))
        print(f"   🦕 DINOv2: {best_dinov2['model_name']} (TEST: {safe_format_number(best_dinov2.get('test_accuracy'), 2)}%)")

    # Overall winner based on TEST ACCURACY
    successful_results = [r for r in results if r['success']]
    if successful_results:
        results_with_test_acc = [r for r in successful_results 
                               if isinstance(r.get('test_accuracy', 0), (int, float))]
        
        if results_with_test_acc:
            overall_best = max(results_with_test_acc, key=lambda x: x.get('test_accuracy', 0))
            overall_fastest = min(successful_results, key=lambda x: x['training_time'])
            
            print(f"\n🏆 OVERALL WINNERS (Based on Test Set Performance):")
            print(f"   🎯 Best Test Accuracy: {overall_best['model_name']} ({safe_format_number(overall_best.get('test_accuracy'), 2)}%) ⭐")
            print(f"   ⚡ Fastest Training: {overall_fastest['model_name']} ({safe_format_number(overall_fastest['training_time'])}s)")
    
    print("\n" + "="*80)
    print("🎉 GPU COMPARISON COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main() 