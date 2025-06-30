# CPU-Optimized Configuration for 100 samples
import multiprocessing as mp

# Dataset Configuration - CPU Optimized
IMAGE_SIZE = (224, 224)  # Smaller than 320x320 for faster CPU processing
BATCH_SIZE = 8  # Smaller batches for CPU
NUM_WORKERS = 0  # Always 0 for CPU training
MAX_WORKERS = 1

# Learning Rate Warmup
USE_WARMUP = False
WARMUP_EPOCHS = 3
WARMUP_START_LR = 1e-6

# Valid file extensions
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Data Split
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# Model Configuration - CPU Friendly
AVAILABLE_MODELS = [
    'efficientnet_b0',  # Lighter than b3
    'efficientnet_b3',  # Heavier but more accurate
    'dinov2_vits14',   # Small DINOv2 variant
]
MODEL_NAME = 'efficientnet_b0'  # Fastest on CPU

# Training Configuration - CPU Optimized
LEARNING_RATE = 0.01  # Higher LR for faster convergence with small data
WEIGHT_DECAY = 0.01
EPOCHS = 20  # More epochs for small dataset
PATIENCE = 5
LR_FACTOR = 0.5
DROPOUT = 0.3

# Use simple heads for faster training
USE_SIMPLE_HEAD = True

# Optimizer Configuration - CPU Optimized
OPTIMIZER = 'sgd'  # SGD is often faster on CPU than AdamW
OPTIMIZER_PARAMS = {
    'sgd': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'momentum': 0.9,
        'nesterov': True
    },
    'adamw': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
}

# Scheduler Configuration
SCHEDULER = 'step'  # Simple step scheduler
SCHEDULER_PARAMS = {
    'step': {
        'step_size': 5,
        'gamma': 0.5
    },
    'plateau': {
        'mode': 'min',
        'patience': PATIENCE,
        'factor': LR_FACTOR
    }
}

# Early Stopping - More patient for small dataset
EARLY_STOPPING_PATIENCE = 8
MIN_DELTA = 0.001
OVERFITTING_THRESHOLD = 0.05  # More lenient for small dataset

# Training Monitoring
TRAIN_JUMP_THRESHOLD = 20
VAL_JUMP_THRESHOLD = 15
DEBUG_BATCHES_INTERVAL = 10  # More frequent for small dataset
DEBUG_FIRST_N_EPOCHS = 3

# Paths
SAVE_DIR = 'models_cpu'

# Simplified Data Augmentation for CPU
TRAIN_TRANSFORMS = {
    'resize': IMAGE_SIZE,
    'horizontal_flip': True,
    'rotation': 15,  # Less rotation for faster processing
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.05
    },
    'normalize': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}

VAL_TEST_TRANSFORMS = {
    'resize': IMAGE_SIZE,
    'normalize': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}

# CPU-Specific Settings
COMPILE_MODEL = False  # Disable model compilation on CPU
USE_GRADIENT_CHECKPOINTING = False  # Not needed for small models
FREEZE_BACKBONE = True  # Keep backbone frozen for faster training

# Perfect for 100 samples - use all of them
FEW_SHOT_MODE = None  # Use all 100 samples
FEW_SHOT_VALUE = None 