# 📝 Model Naming Convention Guide

## Overview
Your saved models now include **experiment context** in their names so you can easily distinguish between different experimental conditions.

## Naming Pattern
```
{original_model_name}_{regime}_{label_percentage}_{seed}
```

## Examples

### DINOv2 Models
- `dinov2_vitb14_LINP_10pct.pth` - Linear probe with 10% labels
- `dinov2_vitb14_FINE_50pct.pth` - Fine-tuning with 50% labels  
- `dinov2_vitb14_FINE_100pct.pth` - Fine-tuning with 100% labels

### CNN Models
- `efficientnet_b4_SUPR_10pct.pth` - Supervised training with 10% labels
- `efficientnet_b4_SUPR_50pct.pth` - Supervised training with 50% labels
- `efficientnet_b4_SUPR_100pct.pth` - Supervised training with 100% labels

### ViT Models
- `vit_base_patch16_224_SUPR_10pct.pth` - Supervised training with 10% labels
- `vit_base_patch16_224_SUPR_50pct.pth` - Supervised training with 50% labels
- `vit_base_patch16_224_SUPR_100pct.pth` - Supervised training with 100% labels

## Regime Codes
- **LINP** = Linear Probe (frozen backbone + linear head)
- **FINE** = Fine-tune (progressive unfreezing)  
- **SUPR** = Supervised (full training from scratch)

## Label Percentages
- **10pct** = 10% of training labels visible
- **50pct** = 50% of training labels visible
- **100pct** = 100% of training labels visible

## Seeds (if using multiple)
- **_s42** = Random seed 42 (default seed omitted for brevity)
- **_s123** = Random seed 123
- **_s456** = Random seed 456

## File Locations
Models are saved in:
- `outputs/checkpoints/` - Primary location
- `models/` - Legacy/backup location

## Quick Commands

### List all saved models:
```bash
python list_saved_models.py
```

### Load a specific model in Python:
```python
import torch

# Load DINOv2 fine-tuned with 50% labels
checkpoint = torch.load('outputs/checkpoints/dinov2_vitb14_FINE_50pct_best_acc87.3.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Check model metadata
print(f"Model: {checkpoint['model_name']}")
print(f"Test accuracy: {checkpoint.get('test_acc', 'N/A')}")
print(f"Training regime: {checkpoint.get('training_regime', 'N/A')}")
print(f"Label budget: {checkpoint.get('label_budget', 'N/A')}")
```

## For Your Thesis
This naming convention allows you to:
1. **Compare models** trained under different label conditions
2. **Analyze performance** across training regimes
3. **Load specific models** for detailed inspection
4. **Create figures** showing model performance vs data availability
5. **Validate results** by reloading and testing saved models

## Example Analysis
```python
# Compare all models trained with 10% labels
models_10pct = [
    'dinov2_vitb14_LINP_10pct.pth',
    'dinov2_vitb14_FINE_10pct.pth', 
    'efficientnet_b4_SUPR_10pct.pth',
    'vit_base_patch16_224_SUPR_10pct.pth'
]

for model_file in models_10pct:
    checkpoint = torch.load(f'outputs/checkpoints/{model_file}')
    print(f"{model_file}: {checkpoint.get('test_acc', 'N/A'):.1f}%")
```
