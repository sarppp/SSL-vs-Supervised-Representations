import torch
import os
from torch import autocast
from torch.amp import GradScaler
from tqdm import tqdm
import datetime
import glob
import re
from .training_utils import is_serializable
from ..config import config_paths


def train_epoch(model, train_loader, criterion, optimizer, device, use_amp=False, epoch=0, config_module=None, training_logger=None):
    """Train for one epoch with gradient accumulation support."""
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        import config as default_config
        config_module = default_config
        # Debug: Check what model actually is
    # print(f"DEBUG: model type = {type(model)}")
    # print(f"DEBUG: model has train method = {hasattr(model, 'train')}")
    # print(f"DEBUG: model = {model}")
    # GRADIENT ACCUMULATION SETUP
    gradient_accum_steps = getattr(config_module, 'GRADIENT_ACCUM_STEPS', 1)
    effective_batch_size = getattr(config_module, 'EFFECTIVE_BATCH_SIZE', config_module.BATCH_SIZE)
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Debug info for first epoch
    if training_logger:
        training_logger.log_debug_info(train_loader, model, epoch)
        if gradient_accum_steps > 1 and epoch == 0:
            training_logger.logger.info(f"Gradient accumulation: {gradient_accum_steps} steps, effective batch: {effective_batch_size}")

    # Create a single GradScaler per epoch (recommended) rather than one per batch
    scaler = GradScaler('cuda', enabled=use_amp and device.type == 'cuda')

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # --------------------------------------------------------------
        # FEW-SHOT SUPPORT: skip batches that contain *no* valid labels
        # (all labels == ignore_index => nothing to learn, avoid scaler error)
        # --------------------------------------------------------------
        valid_mask = labels != -1
        if valid_mask.sum() == 0:
            # Optionally log the skipped batch for debug purposes
            if training_logger and batch_idx == 0 and epoch == 0:
                training_logger.logger.info("WARNING: Skipping batch with no labeled samples (few-shot)")
            continue

        images = images[valid_mask]
        labels = labels[valid_mask]

        # Debug first batch of first epoch
        if training_logger:
            training_logger.log_batch_debug(images, labels, epoch, batch_idx)

        # GRADIENT ACCUMULATION: Only zero gradients at start of accumulation cycle
        if batch_idx % gradient_accum_steps == 0:
            optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            # Scale loss by accumulation steps for correct gradient averaging
            scaled_loss = loss / gradient_accum_steps
            scaler.scale(scaled_loss).backward()
            
            # Only step optimizer every N accumulation steps
            if (batch_idx + 1) % gradient_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Scale loss by accumulation steps for correct gradient averaging
            scaled_loss = loss / gradient_accum_steps
            scaled_loss.backward()
            
            # Only step optimizer every N accumulation steps
            if (batch_idx + 1) % gradient_accum_steps == 0:
                optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        
        # Only count labeled samples (ignore -1 labels for accuracy)
        valid_mask = labels != -1
        total += valid_mask.sum().item()
        correct += ((predicted == labels) & valid_mask).sum().item()

        # Monitor progress every N batches (configurable)
        debug_batches_interval = getattr(config_module, "DEBUG_BATCHES_INTERVAL", 10)
        debug_first_n_epochs = getattr(config_module, "DEBUG_FIRST_N_EPOCHS", 1)
        if batch_idx % debug_batches_interval == 0 and batch_idx > 0:
            current_acc = 100. * correct / total if total > 0 else 0.0
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
            'Loss': f'{loss.item():.4f}',  # Display original loss (not scaled)
            'Acc': f'{100.*correct/total:.1f}%' if total > 0 else 'N/A (no labels)',
            'EffBatch': f'{effective_batch_size}' if gradient_accum_steps > 1 else ''
        })

    final_acc = 100. * correct / total if total > 0 else 0.0
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
            
            # Only count labeled samples (ignore -1 labels for accuracy)
            valid_mask = labels != -1
            total += valid_mask.sum().item()
            correct += ((predicted == labels) & valid_mask).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.1f}%' if total > 0 else 'N/A (no labels)'
            })

    final_acc = 100. * correct / total if total > 0 else 0.0
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
    """Move intermediate checkpoint files to outputs/checkpoints and clean up models folder."""
    if training_logger:
        training_logger.log_cleanup_start()
    
    # Import here to avoid circular imports
    from ..models.model_setup import get_model_identifier
    from ..config import config_paths
    import shutil
    
    model_id = get_model_identifier(model_name, config_module)

    save_dir = getattr(config_module, "SAVE_DIR", None)
    if save_dir is None:
        raise AttributeError("The config_module does not have a SAVE_DIR attribute.")
    
    # Ensure checkpoints directory exists
    os.makedirs(config_paths.CHECKPOINTS_DIR, exist_ok=True)
        
    # Build glob patterns without timestamps
    checkpoint_pattern = os.path.join(save_dir, f"checkpoint_*_{model_id}_*.pth")
    best_pattern = os.path.join(save_dir, f"best_{model_id}_*.pth")
    final_pattern = os.path.join(save_dir, f"final_{model_id}_*.pth")

    # Move intermediate checkpoints to outputs/checkpoints instead of deleting
    for file_path in glob.glob(checkpoint_pattern):
        try:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(config_paths.CHECKPOINTS_DIR, filename)
            shutil.move(file_path, dest_path)
            if training_logger:
                training_logger.log_file_moved(filename, "outputs/checkpoints")
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
            # Updated regex to handle filenames without timestamps
            match = re.search(r'_acc([0-9.]+).pth', f)
            if match:
                acc = float(match.group(1))
                if acc > best_acc:
                    best_acc = acc
                    best_file = f
        for f in best_files:
            if f != best_file:
                try:
                    # Move old best files to checkpoints instead of deleting
                    filename = os.path.basename(f)
                    dest_path = os.path.join(config_paths.CHECKPOINTS_DIR, filename)
                    shutil.move(f, dest_path)
                    if training_logger:
                        training_logger.log_file_moved(f"old best: {filename}", "outputs/checkpoints")
                except Exception as e:
                    if training_logger:
                        training_logger.log_file_deletion_error(f, str(e))
        if best_file:
            if training_logger:
                training_logger.log_file_kept(os.path.basename(best_file), "best model")
    else:
        # Move ALL best files to checkpoints
        for f in best_files:
            try:
                filename = os.path.basename(f)
                dest_path = os.path.join(config_paths.CHECKPOINTS_DIR, filename)
                shutil.move(f, dest_path)
                if training_logger:
                    training_logger.log_file_moved(f"best: {filename}", "outputs/checkpoints")
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
            # Updated regex to handle filenames without timestamps
            match = re.search(r'_acc([0-9.]+).pth', f)
            if match:
                acc = float(match.group(1))
                if acc > best_acc:
                    best_acc = acc
                    best_file = f
        for f in final_files:
            if f != best_file:
                try:
                    # Move old final files to checkpoints instead of deleting
                    filename = os.path.basename(f)
                    dest_path = os.path.join(config_paths.CHECKPOINTS_DIR, filename)
                    shutil.move(f, dest_path)
                    if training_logger:
                        training_logger.log_file_moved(f"old final: {filename}", "outputs/checkpoints")
                except Exception as e:
                    if training_logger:
                        training_logger.log_file_deletion_error(f, str(e))
        if best_file:
            if training_logger:
                training_logger.log_file_kept(os.path.basename(best_file), "final model")
    else:
        # Move ALL final files to checkpoints
        for f in final_files:
            try:
                filename = os.path.basename(f)
                dest_path = os.path.join(config_paths.CHECKPOINTS_DIR, filename)
                shutil.move(f, dest_path)
                if training_logger:
                    training_logger.log_file_moved(f"final: {filename}", "outputs/checkpoints")
            except Exception as e:
                if training_logger:
                    training_logger.log_file_deletion_error(f, str(e))