#!/usr/bin/env python3
"""
Model Renamer - Add Experiment Context to Model Names
========================================================
This script renames your saved models to include experiment context
so you can easily identify which experiment each model came from.
"""
import os
import glob
import json
from pathlib import Path
import shutil

def parse_experiment_from_logs(model_id):
    """Extract experiment context from log files"""
    logs_dir = Path("outputs/logs")
    
    # Look for training log
    training_log = logs_dir / f"training_{model_id}.log"
    
    context = {
        'regime': 'unknown',
        'labels': 'unknown', 
        'accuracy': 'unknown'
    }
    
    if training_log.exists():
        try:
            with open(training_log, 'r') as f:
                content = f.read()
                
                # Extract few-shot info
                if "10 labeled samples per class" in content:
                    context['labels'] = '10pct'
                elif "50 labeled samples per class" in content:
                    context['labels'] = '50pct'
                elif "FEW-SHOT LEARNING MODE ACTIVE" not in content:
                    context['labels'] = '100pct'
                elif "Value: 10" in content:
                    context['labels'] = '10pct'
                elif "Value: 50" in content:
                    context['labels'] = '50pct'
                    
                # Extract training regime from backbone freeze info
                if "Backbone will unfreeze after epoch 999" in content:
                    context['regime'] = 'LINP'  # Linear probe
                elif "Backbone will unfreeze after epoch" in content:
                    context['regime'] = 'FINE'  # Fine-tune
                else:
                    context['regime'] = 'SUPR'  # Supervised
                    
        except Exception as e:
            print(f"WARNING: Could not parse {training_log}: {e}")
    
    return context

def rename_models_with_context():
    """Rename models to include experiment context"""
    
    models_dir = Path("models")
    outputs_models_dir = Path("outputs/checkpoints")
    
    # Check both locations
    locations = []
    if models_dir.exists():
        locations.append(models_dir)
    if outputs_models_dir.exists():
        locations.append(outputs_models_dir)
    
    if not locations:
        print(" No model directories found")
        return
    
    renamed_count = 0
    
    print("RENAMING MODELS WITH EXPERIMENT CONTEXT")
    print("=" * 60)
    
    for location in locations:
        print(f"\n📁 Processing: {location}")
        print("-" * 40)
        
        # Find all .pth files
        model_files = list(location.glob("*.pth"))
        
        for model_file in model_files:
            filename = model_file.name
            
            # Skip if already renamed
            if '_LINP_' in filename or '_FINE_' in filename or '_SUPR_' in filename:
                print(f"Skipping (already renamed): {filename}")
                continue
            
            # Extract model identifier
            model_id = None
            if filename.startswith('best_'):
                # e.g., best_dino_vitb14_label_10_acc32.00.pth
                parts = filename.replace('best_', '').replace('.pth', '').split('_')
                if len(parts) >= 2:
                    if 'dino' in parts[0]:
                        model_id = f"dinov2_{parts[1]}"
                    elif 'cnn' in parts[0]:
                        model_id = "efficientnet_b4"
                    elif 'vit' in parts[0]:
                        model_id = "vit_base_patch16_224"
            
            if not model_id:
                print(f"WARNING: Could not parse model ID from: {filename}")
                continue
            
            # Get experiment context
            context = parse_experiment_from_logs(model_id)
            
            # Extract accuracy from filename
            acc_str = "unknown"
            if "_acc" in filename:
                try:
                    acc_part = filename.split("_acc")[1].replace(".pth", "")
                    acc_num = float(acc_part)
                    acc_str = f"acc{acc_num:.1f}"
                except:
                    acc_str = "unknown"
            
            # Create new filename
            if 'dinov2' in model_id:
                base_name = 'dinov2_vitb14'
            elif 'efficientnet' in model_id:
                base_name = 'efficientnet_b4'
            elif 'vit' in model_id:
                base_name = 'vit_base_patch16_224'
            else:
                base_name = model_id
            
            new_filename = f"{base_name}_{context['regime']}_{context['labels']}_best_{acc_str}.pth"
            new_path = location / new_filename
            
            # Rename the file
            try:
                shutil.move(str(model_file), str(new_path))
                print(f"{filename}")
                print(f"   → {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f" Failed to rename {filename}: {e}")
    
    print(f"\nSUMMARY")
    print("-" * 40)
    print(f"Renamed {renamed_count} model files")
    print(f"Models now include experiment context in filenames")
    
    if renamed_count > 0:
        print(f"\nExamples of new naming:")
        print(f"   dinov2_vitb14_LINP_10pct_best_acc25.3.pth")
        print(f"   dinov2_vitb14_FINE_50pct_best_acc48.0.pth") 
        print(f"   efficientnet_b4_SUPR_100pct_best_acc42.7.pth")

if __name__ == "__main__":
    rename_models_with_context()
