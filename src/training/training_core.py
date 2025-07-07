import torch
import os
import datetime
import json
from ..utils.logger_manager import TrainingLogger
from . import training_utils
from .training_utils import EarlyStopping, monitor_overfitting, monitor_training_stability
from .training_epochs import train_epoch, validate_epoch, save_checkpoint, cleanup_checkpoint_files
from ..config import config_paths


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                model_name=None, class_names=None, class_to_idx=None, use_amp=False, config_module=None, test_loader=None):
    """
    Train the model with comprehensive logging and monitoring.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        model_name: Name of the model for logging
        class_names: List of class names
        class_to_idx: Class name to index mapping
        use_amp: Whether to use Automatic Mixed Precision
        config_module: Configuration module
        test_loader: Optional test data loader (for logging dataset size only)
    """
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        from ..config import config as default_config
        config_module = default_config
        
    model_name = model_name or getattr(config_module, 'MODEL_NAME', 'efficientnet_b4')
    
    # Apply few-shot adjustments if enabled
    few_shot_mode = getattr(config_module, 'FEW_SHOT_MODE', None)
    if few_shot_mode is not None:
        # Adjust epochs for few-shot learning
        original_epochs = getattr(config_module, 'EPOCHS', 15)
        setattr(config_module, 'EPOCHS', min(50, original_epochs * 2))  # Increase epochs but cap at 50
        
        # Adjust early stopping patience
        early_stopping_patience = getattr(config_module, 'EARLY_STOPPING_PATIENCE', 3)
        setattr(config_module, 'EARLY_STOPPING_PATIENCE', max(10, early_stopping_patience * 2))
    
    # Create unique timestamp for this training session
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create save directory if it doesn't exist
    save_dir = getattr(config_module, 'SAVE_DIR', 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup logging with new TrainingLogger
    training_logger = TrainingLogger(model_name, timestamp)
    
    # Prepare config dict and dataset info
    config_dict = {k: v for k, v in config_module.__dict__.items() 
                   if not k.startswith('__') and not callable(v)}
    total_samples = {
        'train': len(train_loader.dataset),
        'val': len(val_loader.dataset),
        'test': len(test_loader.dataset) if test_loader is not None else 0
    }
    
    # Log training start
    training_logger.log_training_start(config_dict, total_samples)
    
    # Apply few-shot adjustments if enabled in config
    few_shot_mode = getattr(config_module, 'FEW_SHOT_MODE', None)
    if few_shot_mode is not None:
        # Adjust epochs for few-shot learning
        original_epochs = getattr(config_module, 'EPOCHS', 15)
        setattr(config_module, 'EPOCHS', min(50, original_epochs * 2))  # Increase epochs but cap at 50
        
        # Adjust early stopping patience
        early_stopping_patience = getattr(config_module, 'EARLY_STOPPING_PATIENCE', 3)
        setattr(config_module, 'EARLY_STOPPING_PATIENCE', max(10, early_stopping_patience * 2))
        
        # Log few-shot configuration
        total_train_samples = len(train_loader.dataset)
        training_logger.log_few_shot_info(
            few_shot_mode,
            getattr(config_module, 'FEW_SHOT_VALUE', 0.1),
            total_train_samples
        )
    
    # Training time tracking
    training_start_time = datetime.datetime.now()
    
    # Initialize tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    best_model_path = None  # Initialize to avoid UnboundLocalError
    
    # For stability monitoring
    prev_train_acc = None
    prev_val_acc = None
    
    # Initialize early stopping (now uses config automatically)
    early_stopping = EarlyStopping(config_module=config_module, training_logger=training_logger)
    
    # Check if this is a DINOv2 model for progressive unfreezing
    is_dinov2 = hasattr(model, 'unfreeze_backbone')
    unfreeze_after = getattr(config_module, 'UNFREEZE_AFTER_EPOCH', 10) if is_dinov2 else None
    backbone_unfrozen = False
    
    # Additional logging for DINOv2 and AMP info
    training_logger.log_amp_status(use_amp)
    
    if is_dinov2:
        training_logger.log_dinov2_detection(unfreeze_after)
    
    epochs = getattr(config_module, 'EPOCHS', 15)
    for epoch in range(epochs):
        training_logger.log_epoch_start(epoch, epochs)
        
        # Progressive unfreezing for DINOv2
        if is_dinov2 and unfreeze_after is not None and epoch >= unfreeze_after and not backbone_unfrozen:
            model.unfreeze_backbone()
            
            # Reduce learning rate when unfreezing (common practice)
            unfreeze_lr_factor = getattr(config_module, 'UNFREEZE_LR_FACTOR', 0.1)
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= unfreeze_lr_factor
                training_logger.log_dinov2_unfreeze(epoch, old_lr, param_group['lr'])
            
            backbone_unfrozen = True
        
        # Training
        training_logger.log_training_phase("training")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_amp, epoch, config_module, training_logger)
        
        # Validation
        training_logger.log_training_phase("validation")
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, use_amp, epoch)
        
        # Update scheduler (handle different scheduler types)
        old_lr = optimizer.param_groups[0]['lr']
        
        # Different schedulers need different step() calls
        scheduler_type = type(scheduler).__name__
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_loss)  # Plateau scheduler needs validation loss
        else:
            scheduler.step()  # Other schedulers (Cosine, Step, Linear, Sequential) don't need arguments
        
        new_lr = optimizer.param_groups[0]['lr']
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Log epoch results and trends
        training_logger.log_epoch_results(epoch, train_loss, train_acc, val_loss, val_acc, new_lr)
        training_logger.log_loss_trends(epoch)
        
        # Log learning rate changes
        if new_lr < old_lr:
            training_logger.log_lr_change(old_lr, new_lr)
        
        # Early stopping warning
        if epoch > 0:
            val_loss_change = val_loss - val_losses[-2]
            if val_loss_change > 0:
                early_stopping_patience = getattr(config_module, 'EARLY_STOPPING_PATIENCE', 10)
                training_logger.log_early_stopping_warning(val_loss_change, early_stopping.counter, early_stopping_patience)
        
        # Monitor stability and unusual behavior
        monitor_training_stability(train_acc, val_acc, epoch, prev_train_acc, prev_val_acc, config_module, training_logger)
        
        # Monitor overfitting (now uses config automatically)
        overfitting_severity = monitor_overfitting(train_acc, val_acc, train_loss, val_loss, config_module=config_module, training_logger=training_logger)
        
        # Additional warnings
        if epoch > 0:
            loss_increase = val_loss > val_losses[-2]
            acc_decrease = val_acc < val_accuracies[-2]
            if loss_increase and acc_decrease:
                training_logger.log_validation_warning(f"📉 WARNING: Both val loss increased and val acc decreased - possible overfitting start")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Import here to avoid circular imports
            from ..models.model_setup import get_model_identifier
            model_id = get_model_identifier(model_name, config_module)
            # Include timestamp to avoid overwrites and more precision for accuracy
            save_dir = getattr(config_module, 'SAVE_DIR', 'models')
            best_model_path = os.path.join(
                save_dir, 
                f'best_{model_id}_acc{val_acc:.2f}.pth'
            )
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, 
                          best_model_path, class_names, class_to_idx, config_module, training_logger)
            training_logger.log_best_model_saved(val_acc, best_model_path)
        
        # Check early stopping (loss-based)
        if early_stopping(val_loss, model, epoch):  # Add epoch parameter
            early_stopping_patience = getattr(config_module, 'EARLY_STOPPING_PATIENCE', 10)
            training_logger.log_early_stopping_triggered(epoch, early_stopping_patience, early_stopping.best_epoch)
            break
            
        # Check severe overfitting early stopping
        if overfitting_severity == "severe":
            training_logger.log_severe_overfitting_stop(train_acc, val_acc)
            break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            image_size = getattr(config_module, 'IMAGE_SIZE', (224, 224))
            img_size_str = f"{image_size[0]}x{image_size[1]}" if isinstance(image_size, tuple) else f"{image_size}"
            # Import here to avoid circular imports
            from ..models.model_setup import get_model_identifier
            model_id = get_model_identifier(model_name, config_module)
            save_dir = getattr(config_module, 'SAVE_DIR', 'models')
            checkpoint_path = os.path.join(
                save_dir, 
                f'checkpoint_ep{epoch+1}_{model_id}_{img_size_str}px.pth'
            )
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, 
                          checkpoint_path, class_names, class_to_idx, config_module, training_logger)
            training_logger.log_checkpoint_saved(checkpoint_path)
        
        # Update previous values for next iteration
        prev_train_acc = train_acc
        prev_val_acc = val_acc
    
    # Final model
    from ..models.model_setup import get_model_identifier
    model_id = get_model_identifier(model_name, config_module)
    save_dir = getattr(config_module, 'SAVE_DIR', 'models')
    final_model_path = os.path.join(
        save_dir, 
        f'final_{model_id}_acc{best_val_acc:.2f}.pth'
    )
    save_checkpoint(model, optimizer, len(train_losses)-1, train_losses[-1], val_losses[-1], 
                  best_val_acc, final_model_path, class_names, class_to_idx, config_module, training_logger)
    
    # Training completion summary
    training_end_time = datetime.datetime.now()
    total_training_time = training_end_time - training_start_time
    
    # Cleanup checkpoint files (pass timestamp as None since it's removed from filenames)
    cleanup_checkpoint_files(timestamp=None, model_name=model_name, keep_best=True, keep_final=False, config_module=config_module, training_logger=training_logger)
    
    # Save training results using TrainingLogger
    results_filename = training_logger.save_training_results(
        best_val_acc, 
        best_model_path if best_model_path else final_model_path, 
        final_model_path, 
        len(train_losses), 
        str(total_training_time)
    )
    
    # Log training completion
    training_logger.log_training_complete(
        best_val_acc, 
        len(train_losses), 
        str(total_training_time), 
        best_model_path if best_model_path else final_model_path, 
        results_filename
    )
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'best_model_path': best_model_path if best_model_path else final_model_path,
        'final_model_path': final_model_path,
        'epochs_trained': len(train_losses),
        'timestamp': timestamp,
        'training_time': str(total_training_time),
        'start_time': training_start_time.isoformat(),
        'end_time': training_end_time.isoformat(),
        'log_file': str(training_logger.log_filename),
        'results_file': results_filename
    } 