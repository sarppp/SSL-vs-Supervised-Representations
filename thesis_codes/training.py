# Main training module - imports from split modules for backward compatibility
# 
# Internal Structure:
# - training_utils.py:  EarlyStopping, monitoring functions
# - training_epochs.py: train_epoch, validate_epoch, checkpointing
# - training_core.py:   main train_model orchestration
#
# For development, you can also import directly from specific modules:
#   from training_utils import EarlyStopping
#   from training_epochs import train_epoch  
#   from training_core import train_model

from training_utils import EarlyStopping, monitor_overfitting, monitor_training_stability, is_serializable
from training_epochs import train_epoch, validate_epoch, save_checkpoint, cleanup_checkpoint_files
from training_core import train_model

# Re-export everything for backward compatibility
__all__ = [
    'EarlyStopping',
    'monitor_overfitting', 
    'monitor_training_stability',
    'is_serializable',
    'train_epoch',
    'validate_epoch', 
    'save_checkpoint',
    'cleanup_checkpoint_files',
    'train_model'
]