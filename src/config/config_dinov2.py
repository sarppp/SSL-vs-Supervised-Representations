import multiprocessing as mp
import config_paths

# Dataset Configuration for DINOv2
IMAGE_SIZE = (224, 224)  # DINOv2 typically works well with 224x224
BATCH_SIZE = 64  # Smaller batch size due to larger models
NUM_WORKERS = 4  # Optimal for 6-core system (fixed Docker shm)
MAX_WORKERS = mp.cpu_count()

# Image Validation (pre-load corruption check)
# Set to False if images are already validated to skip expensive checks
VALIDATE_IMAGES = False

# Learning Rate Warmup (more important for DINOv2)
USE_WARMUP = True
WARMUP_EPOCHS = 5
WARMUP_START_LR = 1e-7

# Valid file extensions
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Data Split
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# Model Configuration
AVAILABLE_MODELS = [
    'dinov2_vits14',  # DINOv2 small (384 dim)
    'dinov2_vitb14',  # DINOv2 base (768 dim)
    'dinov2_vitl14',  # DINOv2 large (1024 dim)
]
MODEL_NAME = 'dinov2_vitb14'  # Default to base model

# DINOv2-specific Training Configuration
LEARNING_RATE = 0.001   # Higher LR for fine-tuning (was too low at 0.0001)
WEIGHT_DECAY = 0.01     # Reduced weight decay
EPOCHS = 15             # More epochs needed for DINOv2 (was only 2!)
PATIENCE = 7            # More patience for LR scheduler
LR_FACTOR = 0.3         # More aggressive LR reduction
DROPOUT = 0.1           # Lower dropout

# Research Paper Settings
USE_SIMPLE_HEAD = True  # Use simple linear heads for fair comparison
# USE_SIMPLE_HEAD = False  # Use advanced MLP heads for best performance

# Model-specific configurations
MODEL_CONFIGS = {
    'dinov2_vits14': {
        'learning_rate': 0.001,   # Increased from 0.0005
        'weight_decay': 0.01,     # Reduced from 0.03
        'dropout': 0.1
    },
    'dinov2_vitb14': {
        'learning_rate': 0.0008,  # Increased from 0.0001
        'weight_decay': 0.01,     # Reduced from 0.05
        'dropout': 0.1
    },
    'dinov2_vitl14': {
        'learning_rate': 0.0005,  # Increased from 0.00005
        'weight_decay': 0.01,     # Reduced from 0.07
        'dropout': 0.15
    }
}

#Frozen: FREEZE_BACKBONE = True, UNFREEZE_AFTER_EPOCH = 999 (never unfreeze)
#Progressive: FREEZE_BACKBONE = True, UNFREEZE_AFTER_EPOCH = 3 (current setting)
#Full Fine-tune: FREEZE_BACKBONE = False (fine-tune from start)

# DINOv2 Training Strategy
FREEZE_BACKBONE = True     # Start frozen for stability, then fine-tune
UNFREEZE_AFTER_EPOCH = 3   # Unfreeze early to allow domain adaptation  
UNFREEZE_LR_FACTOR = 0.1   # Reduce LR when unfreezing

# Optimizer Configuration (DINOv2 often works better with AdamW)
OPTIMIZER = 'adamw'
OPTIMIZER_PARAMS = {
    'adamw': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'betas': (0.9, 0.95),  # Different betas for self-supervised
        'eps': 1e-8
    },
    'adam': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'betas': (0.9, 0.95),
        'eps': 1e-8
    },
    'sgd': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'momentum': 0.9,
        'nesterov': True
    }
}

# Scheduler Configuration (Plateau better for few-shot DINOv2)
SCHEDULER = 'plateau'
SCHEDULER_PARAMS = {
    'plateau': {
        'mode': 'min',
        'patience': PATIENCE,
        'factor': LR_FACTOR
    },
    'cosine': {
        'T_max': EPOCHS,
        'eta_min': 1e-7
    },
    'cosinerestarts': {
        'T_0': 10,
        'T_mult': 2,
        'eta_min': 1e-7
    },
    'step': {
        'step_size': 15,
        'gamma': 0.3
    },
    'linear': {
        'start_factor': 1.0,
        'end_factor': 0.1,
        'total_iters': EPOCHS
    }
}

# Early Stopping (more patience for DINOv2)
EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 0.0005
OVERFITTING_THRESHOLD = 0.05  # More tolerant for self-supervised

# Training Monitoring
TRAIN_JUMP_THRESHOLD = 20
VAL_JUMP_THRESHOLD = 15
DEBUG_BATCHES_INTERVAL = 100
DEBUG_FIRST_N_EPOCHS = 3

# Paths
SAVE_DIR = 'models'

# DINOv2-optimized Data Augmentation (lighter augmentation)
TRAIN_TRANSFORMS = {
    'resize': IMAGE_SIZE,
    'horizontal_flip': True,
    'vertical_flip': 0.2,      # Less aggressive
    'rotation': 10,            # Smaller rotation
    'affine': {
        'translate': (0.1, 0.1),  # Less translation
        'scale': (0.9, 1.1)       # Less scaling
    },
    'color_jitter': {
        'brightness': 0.2,        # Lighter augmentation
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.05
    },
    'normalize': {
        'mean': [0.485, 0.456, 0.406],  # Standard ImageNet normalization
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

# DINOv2 specific settings
USE_GRADIENT_CHECKPOINTING = True  # Save memory for larger models
MIXED_PRECISION = True

COMPILE_MODEL = False  # Set to True for PyTorch 2.0+ speedup 
COMPILE_MODE = 'default'  # Options: 'default', 'reduce-overhead', 'max-autotune'

# Few-Shot Learning
FEW_SHOT_MODE = None  # Options: None, 'percentage', 'per_class'
FEW_SHOT_VALUE = 0.1  # 0.1 = 10%, 0.01 = 1%, or samples per class