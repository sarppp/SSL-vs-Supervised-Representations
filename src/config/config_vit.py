import multiprocessing as mp
import config_paths

# Dataset Configuration for ViT
IMAGE_SIZE = (224, 224)  # Standard ViT input size
BATCH_SIZE = 64
NUM_WORKERS = 4  # Optimal for 6-core system (fixed Docker shm)
MAX_WORKERS = mp.cpu_count()

# Image Validation (pre-load corruption check)
# Set to False if images are already validated to skip expensive checks
VALIDATE_IMAGES = False

# Learning Rate Warmup (important for ViT)
USE_WARMUP = True
WARMUP_EPOCHS = 5
WARMUP_START_LR = 1e-6

# Valid file extensions
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Data Split
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# Model Configuration
AVAILABLE_MODELS = [
    'vit_B',    # ViT Base 16x16 patches
    'vit_S',   # ViT Small 16x16 patches
    'vit_L',   # ViT Large 16x16 patches
    'vit_T',    # ViT Tiny 16x16 patches
]
MODEL_NAME = 'vit_B'  # Default model

# ViT-specific Training Configuration
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
EPOCHS = 15  # Match other models for fair comparison
PATIENCE = 5  # For LR scheduler
LR_FACTOR = 0.5
DROPOUT = 0.1

# Model-specific configurations
MODEL_CONFIGS = {
    'vit_T': {
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'dropout': 0.1
    },
    'vit_S': {
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'dropout': 0.1
    },
    'vit_B': {
        'learning_rate': 0.0008,
        'weight_decay': 0.01,
        'dropout': 0.1
    },
    'vit_L': {
        'learning_rate': 0.0005,
        'weight_decay': 0.015,
        'dropout': 0.15
    }
}

# ViT Training Strategy - similar to DINOv2
FREEZE_BACKBONE = True
UNFREEZE_AFTER_EPOCH = 3   # Unfreeze early for domain adaptation
UNFREEZE_LR_FACTOR = 0.1   # Reduce LR when unfreezing

# Research Paper Settings
USE_SIMPLE_HEAD = True  # Use simple linear heads for fair comparison
# USE_SIMPLE_HEAD = False  # Use advanced MLP heads for best performance

# Optimizer Configuration (ViT works well with AdamW)
OPTIMIZER = 'adamw'
OPTIMIZER_PARAMS = {
    'adamw': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    },
    'adam': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    },
    'sgd': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'momentum': 0.9,
        'nesterov': True
    }
}

# Scheduler Configuration
SCHEDULER = 'plateau'
SCHEDULER_PARAMS = {
    'plateau': {
        'mode': 'min',
        'patience': PATIENCE,
        'factor': LR_FACTOR
    },
    'cosine': {
        'T_max': EPOCHS,
        'eta_min': 1e-6
    },
    'step': {
        'step_size': 7,
        'gamma': 0.1
    },
    'linear': {
        'start_factor': 1.0,
        'end_factor': 0.1,
        'total_iters': EPOCHS
    }
}

# Early Stopping & Overfitting Prevention
EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 0.001
OVERFITTING_THRESHOLD = 0.02

# Training Monitoring & Debugging
TRAIN_JUMP_THRESHOLD = 15
VAL_JUMP_THRESHOLD = 10
DEBUG_BATCHES_INTERVAL = 100
DEBUG_FIRST_N_EPOCHS = 2

# Paths
SAVE_DIR = 'models'

# ViT-optimized Data Augmentation
TRAIN_TRANSFORMS = {
    'resize': IMAGE_SIZE,
    'horizontal_flip': True,
    'vertical_flip': 0.2,
    'rotation': 15,
    'affine': {
        'translate': (0.1, 0.1),
        'scale': (0.9, 1.1)
    },
    'color_jitter': {
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': 0.2,
        'hue': 0.1
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

# ViT specific settings
USE_GRADIENT_CHECKPOINTING = True  # Save memory for larger models
MIXED_PRECISION = True

# Model Compilation (PyTorch 2.0+ speedup)
COMPILE_MODEL = False
COMPILE_MODE = 'default'

# Few-Shot Learning
FEW_SHOT_MODE = None  # Options: None, 'percentage', 'per_class'
FEW_SHOT_VALUE = 0.1  # 0.1 = 10%, 0.01 = 1%, or samples per class
