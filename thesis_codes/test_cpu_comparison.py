#!/usr/bin/env python3
"""
Compare CNN vs DINOv2 on CPU with 100 samples
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import time

# Add thesis_codes to path
sys.path.append(os.path.dirname(__file__))

import config_cpu as config
import training
from model_setup import create_model

# Force CPU usage
device = torch.device('cpu')
print(f"🖥️  Using device: {device}")

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
        num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS
    )
    
    return train_loader, val_loader

def test_model(model_name, train_loader, val_loader, config):
    """Test a specific model"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING {model_name.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Create model using your model_setup.py
        model = create_model(
            num_classes=3, 
            model_name=model_name, 
            config=config
        ).to(device)
        
        # Create optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        
        if config.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), 
                **config.OPTIMIZER_PARAMS['sgd']
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(), 
                **config.OPTIMIZER_PARAMS['adamw']
            )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            **config.SCHEDULER_PARAMS['step']
        )
        
        print(f"⚙️  Model Configuration:")
        print(f"   Model: {model_name}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Create class mappings
        class_names = ['red_class', 'green_class', 'blue_class']
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        print(f"🚀 Starting training...")
        
        # Run training
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
            use_amp=False,  # No mixed precision on CPU
            config_module=config
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"✅ {model_name} training completed!")
        print(f"⏱️  Training time: {training_time:.1f} seconds")
        print(f"📊 Final Results:")
        for key, value in result.items():
            if key not in ['log_file', 'results_file']:  # Skip file paths
                print(f"   {key}: {value}")
        
        return {
            'model_name': model_name,
            'training_time': float(training_time),  # Ensure it's a number
            'success': True,
            'best_val_accuracy': result.get('best_val_acc', 0),  # Map correct key
            'final_train_accuracy': result.get('train_accuracies', [0])[-1] if result.get('train_accuracies') else 0,
            'final_val_accuracy': result.get('val_accuracies', [0])[-1] if result.get('val_accuracies') else 0,
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
            'training_time': float(training_time),  # Ensure it's a number
            'success': False,
            'error': str(e)
        }

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
    print("🥊 CNN vs DINOv2 COMPARISON - 100 SAMPLES ON CPU")
    print("="*80)
    
    # Create synthetic dataset
    X, y = create_synthetic_dataset(
        num_samples=100, 
        num_classes=3, 
        image_size=config.IMAGE_SIZE
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(X, y, config)
    
    # Models to test
    models_to_test = [
        'efficientnet_b3',  # CNN
        'dinov2_vits14',    # DINOv2
    ]
    
    results = []
    
    for model_name in models_to_test:
        # Update config for current model
        config.MODEL_NAME = model_name
        
        # Test the model
        result = test_model(model_name, train_loader, val_loader, config)
        results.append(result)
        
        # Small break between models
        time.sleep(2)
    
    # Print comparison summary
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    for result in results:
        model_name = result['model_name']
        if result['success']:
            print(f"\n🏆 {model_name.upper()}:")
            print(f"   ⏱️  Training Time: {safe_format_number(result['training_time'])}s")
            print(f"   🎯 Best Val Accuracy: {safe_format_number(result.get('best_val_accuracy'), 2)}%")
            print(f"   📈 Final Train Accuracy: {safe_format_number(result.get('final_train_accuracy'), 2)}%")
            print(f"   📉 Final Val Accuracy: {safe_format_number(result.get('final_val_accuracy'), 2)}%")
        else:
            print(f"\n💥 {model_name.upper()}: FAILED")
            print(f"   ❌ Error: {result.get('error', 'Unknown')}")
            print(f"   ⏱️  Time before failure: {safe_format_number(result['training_time'])}s")
    
    # Determine winner
    successful_results = [r for r in results if r['success']]
    if len(successful_results) >= 2:
        # Filter results with numeric accuracy values for comparison
        results_with_acc = [r for r in successful_results 
                           if isinstance(r.get('best_val_accuracy', 0), (int, float))]
        
        if results_with_acc:
            best_model = max(results_with_acc, key=lambda x: x.get('best_val_accuracy', 0))
            fastest_model = min(successful_results, key=lambda x: x['training_time'])
            
            print(f"\n🏅 WINNERS:")
            print(f"   🎯 Best Accuracy: {best_model['model_name']} ({safe_format_number(best_model.get('best_val_accuracy'), 2)}%)")
            print(f"   ⚡ Fastest Training: {fastest_model['model_name']} ({safe_format_number(fastest_model['training_time'])}s)")
        else:
            print(f"\n🏅 Both models completed, but accuracy comparison not available")
    elif len(successful_results) == 1:
        print(f"\n🏅 Only {successful_results[0]['model_name']} completed successfully")
    
    print("\n" + "="*80)
    print("🎉 COMPARISON COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main() 