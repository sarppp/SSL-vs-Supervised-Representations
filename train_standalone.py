#!/usr/bin/env python3
"""
🌾 Crop Pest Detection - Standalone Training Script
===================================================

This script replicates the entire notebook workflow:
1. Data splitting from clean dataset
2. DataLoader creation with transforms  
3. Class weight calculation
4. Model setup and training
5. Test evaluation

Usage:
    python train_standalone.py [--config-overrides]
    python train_standalone.py --epochs 50 --batch-size 64 --model efficientnet_b5
    
    # CNN with CNN-optimized settings
    python train_standalone.py --model efficientnet_b3 --epochs 20

    # DINOv2 with DINOv2-optimized settings  
    python train_standalone.py --model dinov2_vitb14 --epochs 20

"""

import sys
import os
import glob
import argparse
import torch
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the src package to set up all paths
import src

# Import config_paths from the new location
from src.config import config_paths

# Import all required modules
try:
    from src.data import data_splitter
    from src.data import dataloader_setup
    from src.utils import class_weights
    from src.models import model_setup
    from src.training import training
    from src.evaluation import evaluation
    from src.config import config
    from src.config import config_dinov2
except ImportError as e:
    print(f"❌ Could not import required modules: {e}")
    print("Make sure the src directory structure is correct.")
    sys.exit(1)


def setup_environment():
    """Setup environment and check requirements."""
    print("🔧 Setting up environment...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available - using CPU (will be slow)")
    
    # Check required files
    required_files = [
        config_paths.CLEAN_DATASET_PICKLE,
        config_paths.BASE_DATA_DIR
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("   Please ensure you have:")
        print(f"   - {config_paths.CLEAN_DATASET_PICKLE} (cleaned dataset)")
        print(f"   - {config_paths.BASE_DATA_DIR}/ (image directory)")
        sys.exit(1)
    
    print("✅ Environment check passed!")


def select_config(model_name):
    """Select appropriate config based on model type."""
    if model_name and model_name.startswith('dinov2'):
        print(f"🦕 Using DINOv2-specific configuration for {model_name}")
        return config_dinov2
    else:
        print(f"🤖 Using standard configuration for {model_name}")
        return config

def override_config(args, cfg):
    """Override config values with command line arguments."""
    if args.epochs:
        cfg.EPOCHS = args.epochs
        print(f"📊 Config override: EPOCHS = {cfg.EPOCHS}")
    
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size
        print(f"📊 Config override: BATCH_SIZE = {cfg.BATCH_SIZE}")
    
    if args.learning_rate:
        cfg.OPTIMIZER_PARAMS[cfg.OPTIMIZER]['lr'] = args.learning_rate
        print(f"📊 Config override: LEARNING_RATE = {cfg.OPTIMIZER_PARAMS[cfg.OPTIMIZER]['lr']}")
    
    if args.model:
        cfg.MODEL_NAME = args.model
        print(f"📊 Config override: MODEL_NAME = {cfg.MODEL_NAME}")
    
    if args.image_size:
        cfg.IMAGE_SIZE = (args.image_size, args.image_size)
        print(f"📊 Config override: IMAGE_SIZE = {cfg.IMAGE_SIZE}")
    
    if args.patience:
        cfg.EARLY_STOPPING_PATIENCE = args.patience
        print(f"📊 Config override: EARLY_STOPPING_PATIENCE = {cfg.EARLY_STOPPING_PATIENCE}")
    
    # Apply model-specific configs if available and not overridden by user
    if hasattr(cfg, 'MODEL_CONFIGS') and cfg.MODEL_NAME in cfg.MODEL_CONFIGS:
        model_cfg = cfg.MODEL_CONFIGS[cfg.MODEL_NAME]
        if not args.learning_rate and 'learning_rate' in model_cfg:
            cfg.OPTIMIZER_PARAMS[cfg.OPTIMIZER]['lr'] = model_cfg['learning_rate']
            print(f"🎯 Auto-config: Learning rate = {model_cfg['learning_rate']} for {cfg.MODEL_NAME}")
        if not args.batch_size and 'batch_size' in model_cfg:
            cfg.BATCH_SIZE = model_cfg['batch_size']
            print(f"🎯 Auto-config: Batch size = {model_cfg['batch_size']} for {cfg.MODEL_NAME}")
        if 'weight_decay' in model_cfg:
            cfg.OPTIMIZER_PARAMS[cfg.OPTIMIZER]['weight_decay'] = model_cfg['weight_decay']
        if 'dropout' in model_cfg:
            cfg.DROPOUT = model_cfg['dropout']


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='🌾 Crop Pest Detection Training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--model', type=str, choices=['efficientnet_b0', 'efficientnet_b3', 'efficientnet_b5', 'resnet50', 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'], 
                       help='Model architecture')
    parser.add_argument('--image-size', type=int, help='Input image size (will be used as size x size)')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    parser.add_argument('--no-amp', action='store_true', help='Disable Automatic Mixed Precision')
    parser.add_argument('--skip-test', action='store_true', help='Skip test evaluation after training')
    parser.add_argument('--test-only', type=str, help='Skip training and only test specified model file')
    
    # Few-shot learning arguments
    parser.add_argument('--few-shot', type=str, choices=['percentage', 'per_class'], help='Enable few-shot learning mode')
    parser.add_argument('--few-shot-value', type=float, help='Few-shot value (0.01=1%, 0.1=10% for percentage; or samples per class)')
    parser.add_argument('--few-shot-seed', type=int, default=42, help='Random seed for few-shot sampling')
    
    args = parser.parse_args()
    
    print("🌾 CROP PEST DETECTION - STANDALONE TRAINING")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Select appropriate config based on model
    cfg = select_config(args.model or config.MODEL_NAME)
    
    # Setup few-shot learning if specified
    if args.few_shot:
        if not args.few_shot_value:
            print("❌ --few-shot-value is required when using --few-shot")
            sys.exit(1)
        cfg.FEW_SHOT_MODE = args.few_shot
        cfg.FEW_SHOT_VALUE = args.few_shot_value
        # Override random seed if provided for few-shot reproducibility
        if args.few_shot_seed != 42:  # Only override if user specified a different seed
            cfg.RANDOM_STATE = args.few_shot_seed
        print(f"\n🎯 FEW-SHOT LEARNING ENABLED:")
        print(f"   Mode: {cfg.FEW_SHOT_MODE}")
        print(f"   Value: {cfg.FEW_SHOT_VALUE}")
        print(f"   Seed: {cfg.RANDOM_STATE}")
        if cfg.FEW_SHOT_MODE == 'percentage':
            print(f"   Using {cfg.FEW_SHOT_VALUE*100:.1f}% of labeled training data (label hiding)")
        else:
            print(f"   Using {cfg.FEW_SHOT_VALUE} labeled samples per class (label hiding)")
    else:
        cfg.FEW_SHOT_MODE = None
    
    # Override config with command line args
    if any(vars(args).values()):
        print("\n📊 Configuration Overrides:")
        override_config(args, cfg)
    
    # Display current configuration
    print(f"\n⚙️  TRAINING CONFIGURATION:")
    print(f"   Model: {cfg.MODEL_NAME}")
    print(f"   Image Size: {cfg.IMAGE_SIZE}")
    print(f"   Batch Size: {cfg.BATCH_SIZE}")
    print(f"   Epochs: {cfg.EPOCHS}")
    print(f"   Learning Rate: {cfg.OPTIMIZER_PARAMS.get(cfg.OPTIMIZER, {}).get('lr', 'N/A')}")
    print(f"   Early Stopping: {cfg.EARLY_STOPPING_PATIENCE}")
    print(f"   AMP: {'Disabled' if args.no_amp else 'Enabled'}")
    print(f"   Save Directory: {cfg.SAVE_DIR}")
    
    # Show few-shot configuration
    if hasattr(cfg, 'FEW_SHOT_MODE') and cfg.FEW_SHOT_MODE is not None:
        print(f"   🎯 Few-shot: {cfg.FEW_SHOT_MODE} ({cfg.FEW_SHOT_VALUE}) - Label Hiding")
    else:
        print(f"   🎯 Few-shot: Disabled")
    
    # Create save directory if it doesn't exist
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    
    # Test-only mode
    if args.test_only:
        print(f"\n🧪 TEST-ONLY MODE")
        print(f"Loading model: {args.test_only}")
        
        # Still need data for testing (no few-shot for test-only mode)
        print("\n📂 STEP 1: Loading and splitting dataset...")
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = data_splitter.split_clean_dataset(
            pickle_path=config_paths.CLEAN_DATASET_PICKLE,
            base_data_dir=config_paths.BASE_DATA_DIR,
            few_shot_mode=None  # Disable few-shot for test-only mode
        )
        
        print("\n🔧 STEP 2: Creating test dataloader...")
        _, _, test_loader, _, _, test_dataset = dataloader_setup.create_dataloaders(
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
            config_module=cfg, run_batch_test=False
        )
        
        print("\n🤖 STEP 3: Loading saved model...")
        model = model_setup.create_model(len(test_dataset.classes), cfg.MODEL_NAME, cfg)
        checkpoint = torch.load(args.test_only)
        model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print("\n🧪 STEP 4: Testing model...")
        test_results = evaluation.comprehensive_test_evaluation(
            model, test_loader, device, test_dataset.classes,
            model_name=f"{cfg.MODEL_NAME}_loaded", config_module=cfg, use_amp=not args.no_amp
        )
        
        print(f"\n✅ TEST COMPLETE!")
        print(f"🎯 Test Accuracy: {test_results['test_accuracy']:.2f}%")
        return
    
    # =============================================================================
    # FULL TRAINING PIPELINE
    # =============================================================================
    
    print("\n📂 STEP 1: Loading and splitting dataset...")
    # Use the new integrated label hiding approach
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = data_splitter.split_clean_dataset_with_config(
        pickle_path=config_paths.CLEAN_DATASET_PICKLE,
        config_module=cfg,
        base_data_dir=config_paths.BASE_DATA_DIR
    )
    
    print("\n🔧 STEP 2: Creating dataloaders...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = dataloader_setup.create_dataloaders(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
        config_module=cfg, run_batch_test=False
    )
    
    print(f"\n✅ Data preparation complete!")
    print(f"📊 Training samples: {len(train_dataset):,}")
    print(f"📊 Validation samples: {len(val_dataset):,}")
    print(f"📊 Test samples: {len(test_dataset):,}")
    print(f"🏷️ Number of classes: {len(train_dataset.classes)}")
    
    # Check if few-shot label hiding was applied
    if hasattr(cfg, 'FEW_SHOT_MODE') and cfg.FEW_SHOT_MODE is not None:
        # Count labeled vs unlabeled samples in training set
        labeled_count = sum(1 for label in train_labels if label != -1)
        unlabeled_count = len(train_labels) - labeled_count
        print(f"🎯 Few-shot label hiding applied:")
        print(f"   ✅ Labeled samples: {labeled_count:,} ({labeled_count/len(train_labels)*100:.1f}%)")
        print(f"   ❌ Unlabeled samples: {unlabeled_count:,} ({unlabeled_count/len(train_labels)*100:.1f}%)")
        print(f"   📝 Note: Training will ignore samples with label = -1")
    
    print("\n⚖️  STEP 3: Calculating class weights...")
    class_weights_tensor = class_weights.calculate_class_weights(train_labels, train_dataset.classes)
    
    print("\n🤖 STEP 4: Creating model and setup training...")
    model = model_setup.create_model(num_classes=len(train_dataset.classes), model_name=cfg.MODEL_NAME, config=cfg)
    model, criterion, optimizer, scheduler, device = model_setup.setup_training(model, class_weights_tensor, cfg)
    
    print("\n🚀 STEP 5: Starting training...")
    results = training.train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        model_name=cfg.MODEL_NAME,
        class_names=train_dataset.classes,
        class_to_idx=train_dataset.class_to_idx,
        use_amp=not args.no_amp,
        config_module=cfg,
        test_loader=test_loader  # Pass test_loader for dataset size logging
    )
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"📊 Epochs trained: {results['epochs_trained']}")
    print(f"⏰ Training time: {results['training_time']}")
    print(f"💾 Best model saved: {results['best_model_path']}")
    
    # Test evaluation (unless skipped)
    if not args.skip_test:
        # Create a shared evaluation logger for both tests
        try:
            from src.utils.logger_manager import EvaluationLogger
            shared_eval_logger = EvaluationLogger(cfg.MODEL_NAME)
        except ImportError:
            print("⚠️  Warning: logger_manager not found, proceeding without shared logger")
            shared_eval_logger = None
        
        print(f"\n🧪 STEP 6: Evaluating on test set...")
        test_results = evaluation.comprehensive_test_evaluation(
            model, test_loader, device, train_dataset.classes,
            model_name=cfg.MODEL_NAME, config_module=cfg, use_amp=not args.no_amp,
            eval_logger=shared_eval_logger
        )
        print(f"🎯 Test accuracy: {test_results['test_accuracy']:.2f}%")
        
        # Also test the saved best model (using the same logger)
        print(f"\n🔄 Testing saved best model...")
        best_model = model_setup.create_model(len(train_dataset.classes), cfg.MODEL_NAME, cfg)
        checkpoint = torch.load(results['best_model_path'])
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model = best_model.to(device)
        
        # Log separator for the second evaluation in the same log file
        if shared_eval_logger is not None:
            shared_eval_logger.logger.info("\n" + "="*60)
            shared_eval_logger.logger.info("🔄 EVALUATING BEST SAVED MODEL")
            shared_eval_logger.logger.info("="*60)
        
        saved_test_results = evaluation.comprehensive_test_evaluation(
            best_model, test_loader, device, train_dataset.classes,
            model_name=f"{cfg.MODEL_NAME}_best_saved", config_module=cfg, use_amp=not args.no_amp,
            eval_logger=shared_eval_logger
        )
        print(f"🎯 Saved best model test accuracy: {saved_test_results['test_accuracy']:.2f}%")
    
    print(f"\n✅ ALL STEPS COMPLETED SUCCESSFULLY! 🎉")
    print(f"📁 Check '{config.SAVE_DIR}' directory for saved models")
    
    # Summary
    print(f"\n📋 TRAINING SUMMARY:")
    print(f"   Model: {config.MODEL_NAME}")
    print(f"   Best Val Acc: {results['best_val_acc']:.2f}%")
    if not args.skip_test:
        print(f"   Test Acc: {test_results['test_accuracy']:.2f}%")
        print(f"   Saved Model Test Acc: {saved_test_results['test_accuracy']:.2f}%")
    print(f"   Training Time: {results['training_time']}")
    print(f"   Epochs: {results['epochs_trained']}")


if __name__ == "__main__":
    main() 