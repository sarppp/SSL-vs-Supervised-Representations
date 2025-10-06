#!/usr/bin/env python3
"""
 List Saved Models - Quick Model Inventory
============================================
This script helps you see all the models you've saved from different experiments.
"""
import os
import glob
from pathlib import Path

def list_saved_models():
    """List all saved models with their experiment context"""
    
    # Check both possible locations
    locations = [
        "outputs/checkpoints",
        "models"
    ]
    
    all_models = []
    
    for location in locations:
        if os.path.exists(location):
            # Find all .pth files
            pattern = os.path.join(location, "*.pth")
            models = glob.glob(pattern)
            
            for model_path in models:
                filename = os.path.basename(model_path)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                
                all_models.append({
                    'filename': filename,
                    'path': model_path,
                    'size_mb': size_mb,
                    'location': location
                })
    
    if not all_models:
        print(" No saved models found in 'outputs/checkpoints' or 'models' directories")
        return
    
    print("="*80)
    print(" SAVED MODELS INVENTORY")
    print("="*80)
    
    # Group by experiment type
    experiment_groups = {}
    
    for model in all_models:
        filename = model['filename']
        
        # Parse experiment info from filename
        if '_LINP_' in filename:
            regime = 'Linear Probe'
        elif '_FINE_' in filename:
            regime = 'Fine-tune'
        elif '_SUPR_' in filename:
            regime = 'Supervised'
        else:
            regime = 'Unknown'
        
        # Extract label percentage
        if '10pct' in filename:
            labels = '10%'
        elif '50pct' in filename:
            labels = '50%'
        elif '100pct' in filename:
            labels = '100%'
        else:
            labels = 'Unknown'
        
        # Extract model type
        if 'dinov2' in filename:
            model_type = 'DINOv2'
        elif 'efficientnet' in filename:
            model_type = 'CNN'
        elif 'vit' in filename:
            model_type = 'ViT'
        else:
            model_type = 'Unknown'
        
        group_key = f"{labels} Labels"
        if group_key not in experiment_groups:
            experiment_groups[group_key] = []
        
        experiment_groups[group_key].append({
            'model_type': model_type,
            'regime': regime,
            'filename': filename,
            'path': model['path'],
            'size_mb': model['size_mb']
        })
    
    # Display grouped results
    for group, models in sorted(experiment_groups.items()):
        print(f"\n{group}")
        print("-" * 50)
        
        for model in models:
            print(f" {model['model_type']} ({model['regime']})")
            print(f"    File: {model['filename']}")
            print(f"    Path: {model['path']}")
            print(f"    Size: {model['size_mb']:.1f} MB")
            print()
    
    print("="*80)
    print(f"Total models found: {len(all_models)}")
    print("Use these model paths for analysis, inference, or further training!")

if __name__ == "__main__":
    list_saved_models()
