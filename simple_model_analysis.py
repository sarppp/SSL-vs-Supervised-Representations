#!/usr/bin/env python3
"""
🔍 Simple Model Analysis (Headless)
==================================
Analyze your trained model without GUI dependencies
"""

import torch
import torch.nn as nn
import os
import sys
import json
from pathlib import Path

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_codes_dir = os.path.join(current_dir, 'thesis_codes')
if thesis_codes_dir not in sys.path:
    sys.path.append(thesis_codes_dir)

try:
    import model_setup
    import config
    print("✅ Modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def analyze_model_architecture(model_path):
    """Analyze the model architecture and parameters."""
    print(f"🔍 ANALYZING MODEL: {model_path}")
    print("=" * 60)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Extract model information
    model_info = {
        'model_name': checkpoint.get('model_name', 'Unknown'),
        'num_classes': len(checkpoint.get('class_names', [])),
        'class_names': checkpoint.get('class_names', []),
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'best_accuracy': checkpoint.get('best_accuracy', 'Unknown'),
        'validation_accuracy': checkpoint.get('validation_accuracy', 'Unknown')
    }
    
    # 🔧 FIX: Infer model name from filename if not in checkpoint
    if model_info['model_name'] == 'Unknown':
        filename = Path(model_path).name.lower()
        if 'efficientnet_b3' in filename or 'cnn_b3' in filename:
            model_info['model_name'] = 'efficientnet_b3'
            print(f"🔧 Inferred model name from filename: efficientnet_b3")
        elif 'efficientnet_b0' in filename:
            model_info['model_name'] = 'efficientnet_b0'
            print(f"🔧 Inferred model name from filename: efficientnet_b0")
        elif 'efficientnet_b4' in filename:
            model_info['model_name'] = 'efficientnet_b4'
            print(f"🔧 Inferred model name from filename: efficientnet_b4")
        else:
            # Default to efficientnet_b3 as that's what your filename suggests
            model_info['model_name'] = 'efficientnet_b3'
            print(f"🔧 Defaulting to efficientnet_b3 based on filename pattern")
    
    print(f"📊 MODEL INFORMATION:")
    print(f"   Model Name: {model_info['model_name']}")
    print(f"   Number of Classes: {model_info['num_classes']}")
    print(f"   Training Epoch: {model_info['epoch']}")
    print(f"   Best Accuracy: {model_info['best_accuracy']}")
    print(f"   Validation Accuracy: {model_info['validation_accuracy']}")
    
    print(f"\n🏷️  CLASS NAMES:")
    for i, class_name in enumerate(model_info['class_names']):
        emoji = "🍅" if 'tomato' in class_name.lower() else "🌾"
        print(f"   {i:2d}: {emoji} {class_name}")
    
    # Recreate model to analyze architecture
    try:
        model = model_setup.create_model(
            model_info['num_classes'], 
            model_info['model_name'], 
            config
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model architecture recreated")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n🔧 MODEL ARCHITECTURE:")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Model Type: {type(model).__name__}")
        
        # Analyze layer structure
        print(f"\n🏗️  LAYER ANALYSIS:")
        conv_layers = []
        linear_layers = []
        other_layers = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
            elif isinstance(module, nn.Linear):
                linear_layers.append((name, module))
            elif len(list(module.children())) == 0:  # Leaf modules
                other_layers.append((name, type(module).__name__))
        
        print(f"   Convolutional Layers: {len(conv_layers)}")
        if conv_layers:
            print(f"     First Conv: {conv_layers[0][0]} - {conv_layers[0][1]}")
            print(f"     Last Conv: {conv_layers[-1][0]} - {conv_layers[-1][1]}")
        
        print(f"   Linear Layers: {len(linear_layers)}")
        if linear_layers:
            for name, layer in linear_layers:
                print(f"     {name}: {layer.in_features} → {layer.out_features}")
        
        print(f"   Other Layers: {len(other_layers)}")
        
        # Test model with dummy input
        print(f"\n🧪 TESTING MODEL:")
        try:
            dummy_input = torch.randn(1, 3, 384, 384)  # Assuming 384x384 input
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            print(f"   ✅ Model forward pass successful")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Output classes: {output.shape[1]}")
            
            # Show sample predictions (softmax)
            probs = torch.softmax(output, dim=1)[0]
            top_5_indices = torch.topk(probs, 5).indices
            print(f"\n🎯 SAMPLE PREDICTION PROBABILITIES:")
            for i, idx in enumerate(top_5_indices):
                class_name = model_info['class_names'][idx] if idx < len(model_info['class_names']) else f"Class_{idx}"
                prob = probs[idx].item()
                print(f"   {i+1}. {class_name}: {prob:.4f}")
                
        except Exception as e:
            print(f"   ❌ Model test failed: {e}")
        
    except Exception as e:
        print(f"❌ Error recreating model: {e}")
    
    # Save analysis results
    try:
        analysis_file = "model_analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"\n💾 Analysis saved to: {analysis_file}")
    except Exception as e:
        print(f"⚠️  Could not save analysis: {e}")

def main():
    """Main analysis function."""
    model_path = "models/best_cnn_b3_acc82.32_20250702_061634.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print(f"🔍 Looking for .pth files...")
        
        # Search for model files
        import glob
        found_models = glob.glob("**/*.pth", recursive=True)
        if found_models:
            print(f"📁 Found model files:")
            for i, model in enumerate(found_models):
                print(f"   {i}: {model}")
            print(f"\n💡 Update the model_path in the script to use one of these files")
        else:
            print(f"❌ No .pth files found in current directory")
        return
    
    analyze_model_architecture(model_path)

if __name__ == "__main__":
    main() 