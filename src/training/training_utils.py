import torch
import types
from ..utils.logger_manager import TrainingLogger


class EarlyStopping:
    def __init__(self, patience=None, min_delta=None, restore_best_weights=True, config_module=None, training_logger=None):
        # Use default config if none provided (for backward compatibility)
        if config_module is None:
            import config as default_config
            config_module = default_config
            
        self.patience = patience or getattr(config_module, "EARLY_STOPPING_PATIENCE", 10)
        self.min_delta = min_delta or getattr(config_module, "MIN_DELTA", 0.001)
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0  # Track which epoch was best
        self.training_logger = training_logger
        
    def __call__(self, val_loss, model, epoch):  # Add epoch parameter
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
                if self.training_logger:
                    self.training_logger.log_early_stopping_restore(self.best_epoch, self.best_loss)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model weights."""
        self.best_weights = model.state_dict().copy()


def monitor_overfitting(train_acc, val_acc, train_loss=None, val_loss=None, threshold=None, config_module=None, training_logger=None):
    """Monitor and warn about overfitting with improved logic."""
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        import config as default_config
        config_module = default_config
        
    threshold = threshold or getattr(config_module, "OVERFITTING_THRESHOLD", 0.05)
    acc_gap = train_acc - val_acc
    
    # Strong overfitting: Large accuracy gap AND val loss higher than train loss
    if acc_gap > threshold * 100:  # Convert to percentage
        if train_loss is not None and val_loss is not None and val_loss > train_loss:
            severity = "severe" if acc_gap > 10 else "moderate"
            if training_logger:
                training_logger.log_overfitting_analysis(severity, train_acc, val_acc, train_loss, val_loss)
            return severity  # Return severity level
        else:
            severity = "none"
            if training_logger:
                training_logger.log_overfitting_analysis(severity, train_acc, val_acc, train_loss, val_loss)
            return severity
    elif val_acc > train_acc + 2:  # Reduced threshold from 5 to 2
        severity = "none"
        if training_logger:
            training_logger.log_overfitting_analysis(severity, train_acc, val_acc, train_loss, val_loss)
        return severity
    elif abs(acc_gap) <= 2:  # Within 2% is normal
        severity = "none"
        if training_logger:
            training_logger.log_overfitting_analysis(severity, train_acc, val_acc, train_loss, val_loss)
        return severity
    else:
        severity = "mild"
        if training_logger:
            training_logger.log_overfitting_analysis(severity, train_acc, val_acc, train_loss, val_loss)
        return severity


def monitor_training_stability(train_acc, val_acc, epoch, prev_train_acc=None, prev_val_acc=None, config_module=None, training_logger=None):
    """Monitor for unusual jumps in accuracy."""
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        import config as default_config
        config_module = default_config
    
    if training_logger:
        training_logger.log_stability_analysis(
            train_acc, val_acc, epoch, prev_train_acc, prev_val_acc,
            getattr(config_module, "TRAIN_JUMP_THRESHOLD", 0.05), 
            getattr(config_module, "VAL_JUMP_THRESHOLD", 0.05)
        )


def is_serializable(v):
    """Check if a value can be serialized to JSON."""
    return not isinstance(v, types.ModuleType) and not callable(v) 