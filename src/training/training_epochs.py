import torch
import os
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
import datetime
import glob
import re
from .training_utils import is_serializable
from ..config import config_paths


def train_epoch(model, train_loader, criterion, optimizer, device, use_amp=False, epoch=0, config_module=None, training_logger=None):
    """Train for one epoch with debugging."""
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        import config as default_config
        config_module = default_config
    
    # Debug: Check what model actually is
    # print(f"🔍 DEBUG: model type = {type(model)}")
    # print(f"🔍 DEBUG: model has train method = {hasattr(model, 'train')}")
    # print(f"🔍 DEBUG: model = {model}")
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Debug info for first epoch
    if training_logger:
        training_logger.log_debug_info(train_loader, model, epoch)

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Debug first batch of first epoch
        if training_logger:
            training_logger.log_batch_debug(images, labels, epoch, batch_idx)

        optimizer.zero_grad()

        if use_amp:
            scaler = GradScaler('cuda')
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Monitor progress every N batches (configurable)
        debug_batches_interval = getattr(config_module, "DEBUG_BATCHES_INTERVAL", 10)
        debug_first_n_epochs = getattr(config_module, "DEBUG_FIRST_N_EPOCHS", 1)
        if batch_idx % debug_batches_interval == 0 and batch_idx > 0:
            current_acc = 100. * correct / total
            if training_logger:
                training_logger.log_batch_progress(
                    batch_idx,
                    loss.item(),
                    current_acc,
                    epoch,
                    debug_batches_interval,
                    debug_first_n_epochs
                )

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.1f}%'
        })

    final_acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, final_acc


def validate_epoch(model, val_loader, criterion, device, use_amp=False, epoch=0):
    """Validate for one epoch with debugging."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            if use_amp:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.1f}%'
            })

    final_acc = 100. * correct / total
    avg_loss = total_loss / len(val_loader)

    return avg_loss, final_acc


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, filepath, class_names=None, class_to_idx=None, config_module=None, training_logger=None):
    """Save model checkpoint with extra metadata for reproducibility."""
    if config_module is None:
        from src.config import config as default_config
        config_module = default_config
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'timestamp': datetime.datetime.now().isoformat(),
        'image_size': getattr(config_module, 'IMAGE_SIZE', None),
        'batch_size': getattr(config_module, 'BATCH_SIZE', None),
        'num_workers': getattr(config_module, 'NUM_WORKERS', None),
        'optimizer_name': getattr(config_module, 'OPTIMIZER', None),
        'optimizer_params': getattr(config_module, 'OPTIMIZER_PARAMS', None),
        'scheduler_name': getattr(config_module, 'SCHEDULER', None),
        'scheduler_params': getattr(config_module, 'SCHEDULER_PARAMS', None),
        'model_name': getattr(config_module, "MODEL_NAME", "unknown_model"),
        'config_dict': {k: v for k, v in config_module.__dict__.items() if not k.startswith('__') and is_serializable(v)},
    }
    
    if class_names is not None:
        checkpoint['class_names'] = class_names
    if class_to_idx is not None:
        checkpoint['class_to_idx'] = class_to_idx
        
    torch.save(checkpoint, filepath)
    if training_logger:
        training_logger.log_checkpoint_saved(filepath)


def cleanup_checkpoint_files(timestamp, model_name, keep_best=True, keep_final=True, config_module=None, training_logger=None):
    """Clean up intermediate checkpoint files, keeping only the best and/or final models."""
    if training_logger:
        training_logger.log_cleanup_start()
    
    # Import here to avoid circular imports
    from ..models.model_setup import get_model_identifier
    model_id = get_model_identifier(model_name, config_module)

    save_dir = getattr(config_module, "SAVE_DIR", None)
    if save_dir is None:
        raise AttributeError("The config_module does not have a SAVE_DIR attribute.")

    checkpoint_pattern = os.path.join(save_dir, f"checkpoint_*_{model_id}_*{timestamp}.pth")
    best_pattern = os.path.join(save_dir, f"best_{model_id}_*{timestamp}.pth")
    final_pattern = os.path.join(save_dir, f"final_{model_id}_*{timestamp}.pth")

    # Delete all intermediate checkpoints
    for file_path in glob.glob(checkpoint_pattern):
        try:
            os.remove(file_path)
            if training_logger:
                training_logger.log_file_deleted(os.path.basename(file_path))
        except Exception as e:
            if training_logger:
                training_logger.log_file_deletion_error(file_path, str(e))

    # Handle best_... files
    best_files = glob.glob(best_pattern)
    if keep_best:
        # Keep only the best best_... file (highest accuracy)
        best_file = None
        best_acc = -1
        for f in best_files:
            match = re.search(r'acc([0-9.]+)', f) # cleanup logic
            if match:
                acc = float(match.group(1))
                if acc > best_acc:
                    best_acc = acc
                    best_file = f
        for f in best_files:
            if f != best_file:
                try:
                    os.remove(f)
                    if training_logger:
                        training_logger.log_file_deleted(f"old best: {os.path.basename(f)}")
                except Exception as e:
                    if training_logger:
                        training_logger.log_file_deletion_error(f, str(e))
        if best_file:
            if training_logger:
                training_logger.log_file_kept(os.path.basename(best_file), "best model")
    else:
        # Delete ALL best files
        for f in best_files:
            try:
                os.remove(f)
                if training_logger:
                    training_logger.log_file_deleted(f"best: {os.path.basename(f)}")
            except Exception as e:
                if training_logger:
                    training_logger.log_file_deletion_error(f, str(e))

    # Handle final_... files
    final_files = glob.glob(final_pattern)
    if keep_final:
        # Keep only the best final_... file (highest accuracy)
        best_file = None
        best_acc = -1
        for f in final_files:
            match = re.search(r'acc([0-9.]+)', f) # cleanup logic
            if match:
                acc = float(match.group(1))
                if acc > best_acc:
                    best_acc = acc
                    best_file = f
        for f in final_files:
            if f != best_file:
                try:
                    os.remove(f)
                    if training_logger:
                        training_logger.log_file_deleted(f"old final: {os.path.basename(f)}")
                except Exception as e:
                    if training_logger:
                        training_logger.log_file_deletion_error(f, str(e))
        if best_file:
            if training_logger:
                training_logger.log_file_kept(os.path.basename(best_file), "final model")
    else:
        # Delete ALL final files
        for f in final_files:
            try:
                os.remove(f)
                if training_logger:
                    training_logger.log_file_deleted(f"final: {os.path.basename(f)}")
            except Exception as e:
                if training_logger:
                    training_logger.log_file_deletion_error(f, str(e)) 