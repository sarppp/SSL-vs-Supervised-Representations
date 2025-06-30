import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import json
import logging
from datetime import datetime

# Add thesis_codes to path for import
sys.path.append(os.path.dirname(__file__))
import training

# 🎯 CHANGE THESE LINES:
use_dino = False
use_few_shot = False

import config
import config_dinov2

if use_dino:
    active_config = config_dinov2
    model_type = "DINOv2"
else:
    active_config = config
    model_type = "CNN"

training.config = active_config  # Patch the config used in training.py

# 🎯 Setup few-shot learning if enabled
if use_few_shot:
    active_config.FEW_SHOT_MODE = 'percentage'
    active_config.FEW_SHOT_VALUE = 0.3  # Use 30% for quick testing
    few_shot_info = f"few_shot_{active_config.FEW_SHOT_MODE}_{active_config.FEW_SHOT_VALUE}"
else:
    active_config.FEW_SHOT_MODE = None
    few_shot_info = "full_data"

# Just print to console - let original files handle their own logging
print("="*60)
print(f"🧪 QUICK TRAINING TEST - {model_type}")
print("="*60)
print(f"Testing with {model_type} config: {active_config.MODEL_NAME}")
print(f"Few-shot enabled: {use_few_shot}")

# Dummy model
class DummyNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16*16*3, num_classes)
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# Dummy data
num_classes = 3
num_samples = 32
X = torch.randn(num_samples, 3, 16, 16)
y = torch.randint(0, num_classes, (num_samples,))

# 🎯 Apply few-shot simulation to dummy data
if use_few_shot:
    from data_splitter import apply_few_shot
    # Convert to lists for few-shot processing
    dummy_paths = [f"dummy_path_{i}" for i in range(num_samples)]
    dummy_labels = [f"class_{label.item()}" for label in y]
    
    print(f"\n🎯 APPLYING FEW-SHOT TO DUMMY DATA:")
    print(f"   Original dummy samples: {num_samples}")
    print(f"   Few-shot mode: {active_config.FEW_SHOT_MODE}")
    print(f"   Few-shot value: {active_config.FEW_SHOT_VALUE}")
    
    # Apply few-shot reduction
    few_shot_paths, few_shot_labels = apply_few_shot(
        dummy_paths, dummy_labels, 
        active_config.FEW_SHOT_MODE, 
        active_config.FEW_SHOT_VALUE
    )
    
    # Convert back to tensors
    few_shot_indices = [dummy_paths.index(path) for path in few_shot_paths]
    X_few_shot = X[few_shot_indices]
    y_few_shot = y[few_shot_indices]
    
    dataset = TensorDataset(X_few_shot, y_few_shot)
    print(f"   Final dataset size: {len(dataset)} samples")
else:
    dataset = TensorDataset(X, y)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(dataset, batch_size=8)

print(f"📊 Test Data Setup:")
print(f"   Number of classes: {num_classes}")
print(f"   Total samples used for training: {len(dataset)}")
print(f"   Input shape: {X.shape}")
print(f"   Batch size: 8")

# Model, loss, optimizer, scheduler
model = DummyNet(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

device = torch.device('cpu')

print(f"⚙️ Configuration Details:")
print(f"   Model: {active_config.MODEL_NAME}")
print(f"   Image Size: {active_config.IMAGE_SIZE}")
print(f"   Batch Size: {active_config.BATCH_SIZE}")
print(f"   Optimizer: {active_config.OPTIMIZER}")
print(f"   Optimizer Params: {active_config.OPTIMIZER_PARAMS[active_config.OPTIMIZER]}")
print(f"   Scheduler: {active_config.SCHEDULER}")
print(f"   Scheduler Params: {active_config.SCHEDULER_PARAMS[active_config.SCHEDULER]}")

print("🚀 Starting training test...")

# Run training - will automatically use config.MODEL_NAME
result = training.train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device,
    class_names=[str(i) for i in range(num_classes)], 
    class_to_idx={str(i): i for i in range(num_classes)}, 
    use_amp=False
)

print("✅ Training test completed!")
print("📊 Training result summary:")
for k, v in result.items():
    print(f"   {k}: {v}")

print("="*60)
print(f"🎉 TEST COMPLETED SUCCESSFULLY!")
print("="*60)

# Check what log files were created by training.py
if 'log_file' in result:
    print(f"\n🎯 TRAINING LOG CREATED:")
    print(f"📄 Log: {result['log_file']}")
if 'results_file' in result:
    print(f"📊 Results: {result['results_file']}")