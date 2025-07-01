import multiprocessing as mp

# Dataset Configuration
IMAGE_SIZE = (320, 320)
BATCH_SIZE = 32
NUM_WORKERS = 0  # Keep at 0 to avoid file access issues
MAX_WORKERS = mp.cpu_count()

# Learning Rate Warmup
USE_WARMUP = False  # Set to True to enable
WARMUP_EPOCHS = 5   # Number of epochs to warm up
WARMUP_START_LR = 1e-6  # Starting LR for warmup


# Valid file extensions
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Data Split
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# Model Configuration
AVAILABLE_MODELS = [
    'efficientnet_b0', 
    'efficientnet_b3', 
    'dinov2_vits14',  # DINOv2 small
    'dinov2_vitb14',  # DINOv2 base
]
MODEL_NAME = 'efficientnet_b3'  # Default model

# Training Configuration
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
EPOCHS = 10
PATIENCE = 3  # For LR scheduler
LR_FACTOR = 0.5
DROPOUT = 0.2

# Research Paper Settings
USE_SIMPLE_HEAD = True  # Use simple linear heads for fair comparison
# USE_SIMPLE_HEAD = False  # Use advanced MLP heads for best performance

# Optimizer Configuration
OPTIMIZER = 'adamw'  # Options: 'adamw', 'adam', 'sgd', 'rmsprop'
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
    },
    'rmsprop': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'momentum': 0.9,
        'alpha': 0.99
    }
}

# Scheduler Configuration
SCHEDULER = 'plateau'  # Options: 'plateau', 'cosine', 'step', 'linear'
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
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 epochs
MIN_DELTA = 0.001  # Minimum improvement to be considered as improvement
OVERFITTING_THRESHOLD = 0.02  # If train_acc - val_acc > 2%, warn about overfitting

# Training Monitoring & Debugging
TRAIN_JUMP_THRESHOLD = 15  # Warn if training accuracy jumps >15% in one epoch
VAL_JUMP_THRESHOLD = 10    # Warn if validation accuracy jumps >10% in one epoch
DEBUG_BATCHES_INTERVAL = 100  # Print debug info every N batches
DEBUG_FIRST_N_EPOCHS = 2     # Show detailed debug for first N epochs

# Paths
SAVE_DIR = 'models'

# Enhanced Data Augmentation
TRAIN_TRANSFORMS = {
    'resize': IMAGE_SIZE,
    'horizontal_flip': True,
    'vertical_flip': 0.3,
    'rotation': 20,
    'affine': {
        'translate': (0.2, 0.2),
        'scale': (0.8, 1.2)
    },
    'color_jitter': {
        'brightness': 0.4,
        'contrast': 0.4,
        'saturation': 0.3,
        'hue': 0.1
    },
    'extra_brightness': 0.3,
    'extra_contrast': 0.3,
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

# Few-Shot Learning  
FEW_SHOT_MODE = 'percentage'  # Options: None, 'percentage', 'per_class'
FEW_SHOT_VALUE = 0.1  # 0.1 = 10%, 0.01 = 1%, or samples per class