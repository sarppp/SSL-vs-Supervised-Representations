import logging
import os
import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import numpy as np

class TrainingLogger:
    """Centralized logging for training sessions with structured output."""
    
    def __init__(self, model_name: str, timestamp: Optional[str] = None):
        self.model_name = model_name
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{model_name}_{self.timestamp}"
        
        # Setup directories
        self.log_dir = Path("logs")
        self.results_dir = Path("training_results")
        self.log_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup loggers
        self.logger = self._setup_logger()
        self.log_filename = self.log_dir / f"training_{self.session_id}.log"
        
        # Metrics tracking
        self.metrics_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # Session metadata
        self.session_start = datetime.datetime.now()
        self.session_metadata = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers."""
        logger = logging.getLogger(f"training_{self.session_id}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        log_file = self.log_dir / f"training_{self.session_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_training_start(self, config_dict: Dict[str, Any], total_samples: Dict[str, int]):
        """Log training session start information."""
        self.session_metadata.update({
            'config': config_dict,
            'total_samples': total_samples,
            'start_time': self.session_start.isoformat()
        })
        
        self.logger.info("🚀 STARTING TRAINING SESSION")
        self.logger.info(f"📊 Max epochs: {config_dict.get('EPOCHS', 'N/A')}")
        self.logger.info(f"🛑 Early stopping patience: {config_dict.get('EARLY_STOPPING_PATIENCE', 'N/A')}")
        self.logger.info(f"📁 Save directory: {config_dict.get('SAVE_DIR', 'N/A')}")
        self.logger.info(f"📋 Batch size: {config_dict.get('BATCH_SIZE', 'N/A')}")
        self.logger.info(f"📐 Image size: {config_dict.get('IMAGE_SIZE', 'N/A')}")
        self.logger.info(f"🔖 Session ID: {self.session_id}")
        self.logger.info(f"⏰ Started at: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"📄 Log file: {self.log_filename}")
        
        # Dataset info
        if total_samples:
            self.logger.info(f"📊 Dataset: Train={total_samples.get('train', 0)}, "
                           f"Val={total_samples.get('val', 0)}, Test={total_samples.get('test', 0)}")
    
    def log_few_shot_info(self, mode: str, value: float, total_train_samples: int):
        """Log few-shot learning configuration."""
        self.logger.info("🎯 FEW-SHOT LEARNING MODE ACTIVE")
        self.logger.info(f"   📊 Mode: {mode}")
        self.logger.info(f"   📈 Value: {value}")
        self.logger.info(f"   🏷️ Training samples being used: {total_train_samples}")
        
        if mode == 'percentage':
            self.logger.info(f"   💡 Using {value*100:.1f}% of labeled training data")
            estimated_total = int(total_train_samples / value)
            self.logger.info(f"   🖼️ Estimated total available images: ~{estimated_total}")
        elif mode == 'per_class':
            self.logger.info(f"   💡 Using {value} labeled samples per class")
        
        self.logger.info("   🔬 This simulates real-world scarce labeling scenario")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.logger.info(f"\n🔄 Epoch {epoch+1}/{total_epochs}")
    
    def log_training_phase(self, phase: str):
        """Log training/validation phase."""
        emoji = "📚" if phase == "training" else "🔍"
        self.logger.info(f"{emoji} {phase.capitalize()}...")
    
    def log_epoch_results(self, epoch: int, train_loss: float, train_acc: float, 
                         val_loss: float, val_acc: float, lr: float):
        """Log epoch results and update metrics."""
        # Store metrics
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['train_losses'].append(train_loss)
        self.metrics_history['val_losses'].append(val_loss)
        self.metrics_history['train_accuracies'].append(train_acc)
        self.metrics_history['val_accuracies'].append(val_acc)
        self.metrics_history['learning_rates'].append(lr)
        
        # Log results
        self.logger.info(f"📊 Train: {train_loss:.4f} | {train_acc:.2f}%")
        self.logger.info(f"📊 Val: {val_loss:.4f} | {val_acc:.2f}%")
        self.logger.info(f"📈 LR: {lr:.6f}")
    
    def log_loss_trends(self, epoch: int):
        """Log loss trend analysis."""
        if epoch == 0:
            return
        
        val_losses = self.metrics_history['val_losses']
        train_losses = self.metrics_history['train_losses']
        
        val_change = val_losses[-1] - val_losses[-2]
        train_change = train_losses[-1] - train_losses[-2]
        
        val_trend = "📈" if val_change > 0 else "📉" if val_change < 0 else "➡️"
        train_trend = "📈" if train_change > 0 else "📉" if train_change < 0 else "➡️"
        
        self.logger.info(f"📊 Loss Trends: Train {train_losses[-2]:.4f}→{train_losses[-1]:.4f} {train_trend} | "
                        f"Val {val_losses[-2]:.4f}→{val_losses[-1]:.4f} {val_trend}")
        
        # Loss gap analysis
        loss_gap = val_losses[-1] - train_losses[-1]
        if loss_gap > 0:
            gap_msg = "Val loss > Train loss (possible overfitting)"
        elif loss_gap < -0.1:
            gap_msg = "Train loss > Val loss (strong regularization)"
        else:
            gap_msg = "Normal"
        self.logger.info(f"📏 Loss Gap: {loss_gap:.4f} ({gap_msg})")
    
    def log_lr_change(self, old_lr: float, new_lr: float):
        """Log learning rate changes."""
        if new_lr < old_lr:
            self.logger.info(f"📉 LR reduced: {old_lr:.6f} → {new_lr:.6f}")
    
    def log_early_stopping_warning(self, val_loss_increase: float, counter: int, patience: int):
        """Log early stopping warnings."""
        self.logger.info(f"⚠️ Val loss increased by {val_loss_increase:.4f} "
                        f"(Early stop counter: {counter}/{patience})")
    
    def log_best_model_saved(self, val_acc: float, filepath: str):
        """Log best model checkpoint."""
        self.logger.info(f"🎉 New best: {val_acc:.2f}% - Saved to {filepath}")
    
    def log_checkpoint_saved(self, filepath: str):
        """Log regular checkpoint save."""
        self.logger.info(f"💾 Checkpoint saved: {filepath}")
    
    def log_overfitting_warning(self, severity: str, train_acc: float, val_acc: float, 
                               train_loss: float = None, val_loss: float = None):
        """Log overfitting warnings."""
        acc_gap = train_acc - val_acc
        
        if severity == "severe":
            self.logger.warning(f"⚠️ OVERFITTING WARNING (SEVERE): Train accuracy ({train_acc:.2f}%) "
                              f"is {acc_gap:.2f}% higher than validation ({val_acc:.2f}%)")
            if train_loss and val_loss:
                self.logger.warning(f"   Loss gap confirms: Train loss ({train_loss:.4f}) < Val loss ({val_loss:.4f})")
        elif severity == "moderate":
            self.logger.warning(f"⚠️ OVERFITTING WARNING (MODERATE): Train accuracy ({train_acc:.2f}%) "
                              f"is {acc_gap:.2f}% higher than validation ({val_acc:.2f}%)")
    
    def log_stability_warning(self, metric_type: str, prev_val: float, curr_val: float, jump: float):
        """Log unusual jumps in metrics."""
        self.logger.warning(f"🚨 UNUSUAL {metric_type.upper()} JUMP: {prev_val:.1f}% → {curr_val:.1f}% "
                          f"({jump:.1f}% jump!)")
        if metric_type == "train":
            self.logger.warning("   This might indicate: Data leakage, wrong transforms, or training issues")
    
    def log_early_stopping_triggered(self, epoch: int, patience: int, best_epoch: int):
        """Log early stopping trigger."""
        self.logger.info(f"\n🛑 EARLY STOPPING triggered after {epoch+1} epochs")
        self.logger.info(f"📉 No improvement in validation loss for {patience} epochs")
        self.logger.info(f"🏆 Using best model from epoch {best_epoch + 1}")
    
    def log_severe_overfitting_stop(self, train_acc: float, val_acc: float):
        """Log severe overfitting early stop."""
        self.logger.warning(f"\n🚨 SEVERE OVERFITTING DETECTED - STOPPING TRAINING")
        self.logger.warning(f"📊 Train accuracy ({train_acc:.1f}%) significantly higher than validation ({val_acc:.1f}%)")
        self.logger.warning(f"🛑 Stopping to prevent further overfitting")
    
    def log_dinov2_unfreeze(self, epoch, old_lr, new_lr):
        """Log DINOv2 backbone unfreezing."""
        self.logger.info(f"🦖 UNFREEZING: DINOv2 backbone unfrozen at epoch {epoch + 1}")
        self.logger.info(f"📉 LR reduced: {old_lr:.2e} → {new_lr:.2e}")

    def log_amp_status(self, use_amp):
        """Log AMP (Automatic Mixed Precision) status."""
        self.logger.info(f"⚡ AMP: {'Enabled' if use_amp else 'Disabled'}")

    def log_dinov2_detection(self, unfreeze_after=None):
        """Log DINOv2 model detection and unfreezing info."""
        self.logger.info(f"🦖 DINOv2 model detected")
        if unfreeze_after is not None:
            self.logger.info(f"🧊 Backbone will unfreeze after epoch {unfreeze_after}")

    def log_validation_warning(self, message):
        """Log general validation warnings."""
        self.logger.warning(message)

    def log_cleanup_start(self):
        """Log start of cleanup process."""
        self.logger.info(f"\n🧹 Cleaning up intermediate checkpoint files...")

    def log_file_deleted(self, filename: str):
        """Log successful file deletion."""
        self.logger.info(f"🗑️ Deleted: {filename}")

    def log_file_deletion_error(self, filepath: str, error: str):
        """Log file deletion error."""
        self.logger.error(f"❌ Failed to delete {filepath}: {error}")

    def log_file_kept(self, filename: str, file_type: str = ""):
        """Log file kept during cleanup."""
        prefix = f"{file_type} " if file_type else ""
        self.logger.info(f"✅ Kept {prefix}file: {filename}")

    def log_cleanup(self, deleted_files: List[str], kept_files: List[str]):
        """Log checkpoint cleanup (legacy method for backward compatibility)."""
        self.log_cleanup_start()
        for file in deleted_files:
            self.logger.info(f"🗑️ Deleted: {file}")
        for file in kept_files:
            self.logger.info(f"✅ Kept: {file}")
    
    def log_training_complete(self, best_val_acc: float, epochs_trained: int, 
                            training_time: str, best_model_path: str, results_file: str):
        """Log training completion summary."""
        session_end = datetime.datetime.now()
        
        self.logger.info(f"\n✅ TRAINING COMPLETE!")
        self.logger.info(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
        self.logger.info(f"📊 Total epochs trained: {epochs_trained}")
        self.logger.info(f"⏰ Training time: {training_time}")
        self.logger.info(f"📅 Started: {self.session_start.strftime('%H:%M:%S')}")
        self.logger.info(f"📅 Finished: {session_end.strftime('%H:%M:%S')}")
        
        if epochs_trained > 0:
            avg_time = (session_end - self.session_start).total_seconds() / epochs_trained
            self.logger.info(f"⚡ Average time per epoch: {avg_time:.1f} seconds")
        
        self.logger.info(f"📄 Log saved: {self.log_filename}")
        self.logger.info(f"📊 Results saved: {results_file}")
    
    def save_training_results(self, best_val_acc: float, best_model_path: str, 
                            final_model_path: str, epochs_trained: int, training_time: str) -> str:
        """Save training results to JSON file."""
        session_end = datetime.datetime.now()
        
        results = {
            'session_id': self.session_id,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'start_time': self.session_start.isoformat(),
            'end_time': session_end.isoformat(),
            'training_time': training_time,
            'epochs_trained': epochs_trained,
            'best_val_acc': best_val_acc,
            'metrics_history': self.metrics_history,
            'session_metadata': self.session_metadata,
            'files': {
                'log_file': str(self.log_filename),
                'best_model': best_model_path,
                'final_model': final_model_path
            }
        }
        
        results_file = self.results_dir / f"training_{self.session_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(results_file)
    
    def log_overfitting_analysis(self, severity: str, train_acc: float, val_acc: float, 
                                train_loss: float = None, val_loss: float = None):
        """Log overfitting analysis from monitor_overfitting function."""
        if severity in ["severe", "moderate"]:
            self.log_overfitting_warning(severity, train_acc, val_acc, train_loss, val_loss)
        elif severity == "none":
            acc_gap = train_acc - val_acc
            if val_acc > train_acc + 2:
                self.logger.info(f"✅ STRONG REGULARIZATION: Val ({val_acc:.1f}%) > Train ({train_acc:.1f}%) - Excellent generalization!")
            elif abs(acc_gap) <= 2:
                self.logger.info(f"📊 Healthy gap: Train ({train_acc:.1f}%) vs Val ({val_acc:.1f}%)")
            else:
                self.logger.info(f"📊 Large accuracy gap ({acc_gap:.1f}%) but loss pattern is normal - likely due to regularization")
        elif severity == "mild":
            self.logger.info(f"📊 Moderate gap: Train ({train_acc:.1f}%) vs Val ({val_acc:.1f}%) - Monitor closely")

    def log_stability_analysis(self, train_acc: float, val_acc: float, epoch: int, 
                              prev_train_acc: float = None, prev_val_acc: float = None, 
                              train_threshold: float = 15.0, val_threshold: float = 10.0):
        """Log stability analysis from monitor_training_stability function."""
        if prev_train_acc is not None:
            train_jump = abs(train_acc - prev_train_acc)
            if train_jump > train_threshold:
                self.log_stability_warning("train", prev_train_acc, train_acc, train_jump)
        
        if prev_val_acc is not None:
            val_jump = abs(val_acc - prev_val_acc)
            if val_jump > val_threshold:
                self.log_stability_warning("val", prev_val_acc, val_acc, val_jump)

    def log_early_stopping_restore(self, best_epoch: int, best_loss: float):
        """Log early stopping model restoration."""
        self.logger.info(f"🔄 Restored model to epoch {best_epoch + 1} weights (best validation loss: {best_loss:.4f})")

    def log_debug_info(self, train_loader, model, epoch: int = 0):
        """Log debug information for training."""
        if epoch == 0:
            self.logger.info(f"🔍 Debug Info:")
            self.logger.info(f"   Training batches: {len(train_loader)}")
            self.logger.info(f"   Batch size: {train_loader.batch_size}")
            self.logger.info(f"   Model training mode: {model.training}")

    def log_batch_debug(self, images, labels, epoch: int, batch_idx: int):
        """Log first batch debug information."""
        if epoch == 0 and batch_idx == 0:
            self.logger.info(f"   First batch shape: {images.shape}")
            self.logger.info(f"   Label range: {labels.min().item()} to {labels.max().item()}")
            self.logger.info(f"   Image range: {images.min().item():.3f} to {images.max().item():.3f}")

    def log_batch_progress(self, batch_idx: int, loss: float, current_acc: float, epoch: int, 
                          debug_interval: int = 100, debug_epochs: int = 3):
        """Log batch progress during training."""
        if batch_idx % debug_interval == 0 and batch_idx > 0 and epoch < debug_epochs:
            self.logger.info(f"   Batch {batch_idx}: Loss={loss:.4f}, Acc={current_acc:.1f}%")

    def get_logger(self) -> logging.Logger:
        """Get the underlying logger for direct use."""
        return self.logger


class EvaluationLogger:
    """Simple logging for evaluation sessions."""
    
    def __init__(self, model_name: str, timestamp: Optional[str] = None):
        self.model_name = model_name
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{model_name}_{self.timestamp}"
        
        # Setup directories
        self.log_dir = Path("logs")
        self.results_dir = Path("evaluation_results")
        self.log_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        self.log_filename = self.log_dir / f"evaluation_{self.session_id}.log"
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers."""
        logger = logging.getLogger(f"evaluation_{self.session_id}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File and console handlers
        log_file = self.log_dir / f"evaluation_{self.session_id}.log"
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_evaluation_start(self, few_shot_enabled: bool, config_dict: Dict[str, Any] = None):
        """Log evaluation session start."""
        self.logger.info(f"\n🧪 EVALUATING MODEL ON TEST SET")
        self.logger.info("=" * 50)
        self.logger.info("⚠️  This is the FIRST TIME the model sees test data!")
        self.logger.info(f"📄 Evaluation log: {self.log_filename}")
        self.logger.info(f"🎯 Few-shot training: {few_shot_enabled}")
        
        if few_shot_enabled and config_dict:
            mode = config_dict.get('FEW_SHOT_MODE')
            value = config_dict.get('FEW_SHOT_VALUE')
            self.logger.info(f"   Mode: {mode}")
            self.logger.info(f"   Value: {value}")
            if mode == 'percentage':
                self.logger.info(f"   Training used {value*100:.1f}% of labeled data")
            else:
                self.logger.info(f"   Training used {value} samples per class")
    
    def log_test_accuracy(self, accuracy: float):
        """Log test accuracy."""
        self.logger.info(f"🎯 TEST ACCURACY: {accuracy:.2f}%")
        self.logger.info("=" * 50)
    
    def log_class_balance_analysis(self, class_names: List[str], true_counts: Dict, pred_counts: Dict):
        """Log class balance analysis."""
        self.logger.info(f"\n📊 CLASS BALANCE ANALYSIS")
        self.logger.info("=" * 50)
        self.logger.info(f"📋 TRUE vs PREDICTED DISTRIBUTION:")
        self.logger.info(f"{'Class':<20} {'True Count':<12} {'Pred Count':<12} {'True %':<10} {'Pred %':<10} {'Difference'}")
        self.logger.info("-" * 80)
        
        total_true = sum(true_counts.values())
        total_pred = sum(pred_counts.values())
        imbalance_detected = False
        
        for i, class_name in enumerate(class_names):
            true_count = true_counts.get(i, 0)
            pred_count = pred_counts.get(i, 0)
            
            true_pct = (true_count / total_true) * 100
            pred_pct = (pred_count / total_pred) * 100
            difference = pred_pct - true_pct
            
            status = ""
            if abs(difference) > 5:
                status = "⚠️" if abs(difference) > 10 else "⚡"
                imbalance_detected = True
            
            self.logger.info(f"{class_name:<20} {true_count:<12} {pred_count:<12} {true_pct:<10.1f} {pred_pct:<10.1f} {difference:>+7.1f}% {status}")
        
        self.logger.info("-" * 80)
        
        if imbalance_detected:
            self.logger.warning("⚠️  PREDICTION IMBALANCE DETECTED!")
            self.logger.warning("   Some classes are over/under-predicted by >5%")
            self.logger.warning("   Consider adjusting class weights or data augmentation")
        else:
            self.logger.info("✅ BALANCED PREDICTIONS!")
            self.logger.info("   Class weights appear to be working well")
    
    def log_per_class_accuracy(self, class_names: List[str], y_true, y_pred):
        """Log per-class accuracy with warnings."""
        self.logger.info(f"\n🎯 PER-CLASS ACCURACY:")
        self.logger.info("-" * 40)
        
        for i, class_name in enumerate(class_names):
            class_mask = (y_true == i)
            class_count = class_mask.sum()
            if class_count > 0:
                class_acc = (y_pred[class_mask] == i).mean() * 100
                status = []
                
                if class_count < 20:
                    status.append("⚠️ Very few samples (unreliable estimate)")
                if class_acc == 100.0 and class_count < 100:
                    status.append("⚠️ Possible overfitting (too perfect, few samples)")
                elif class_acc > 98.0 and class_count < 50:
                    status.append("⚠️ Likely overfitting (very high, very few samples)")
                if class_acc < 70.0:
                    status.append("❗ Underfitting or class confusion")
                
                status_str = " ".join(status)
                self.logger.info(f"{class_name:<25}: {class_acc:>6.1f}% ({class_count} samples) {status_str}")
    
    def log_evaluation_summary(self, model_name: str, test_accuracy: float, total_samples: int, 
                              class_names: List[str], few_shot_enabled: bool, config_dict: Dict = None, save_dir: str = None):
        """Log evaluation summary."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🏁 TEST EVALUATION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"🤖 Model: {model_name}")
        self.logger.info(f"🎯 Test Accuracy: {test_accuracy:.2f}%")
        self.logger.info(f"🔬 Few-shot training: {few_shot_enabled}")
        
        if few_shot_enabled and config_dict:
            mode = config_dict.get('FEW_SHOT_MODE')
            value = config_dict.get('FEW_SHOT_VALUE')
            if mode == 'percentage':
                self.logger.info(f"   Trained with {value*100:.1f}% labeled data")
            else:
                self.logger.info(f"   Trained with {value} samples/class")
        
        self.logger.info(f"📊 Total Test Samples: {total_samples}")
        self.logger.info(f"🏷️  Number of Classes: {len(class_names)}")
        if save_dir:
            self.logger.info(f"💾 Results saved to: {save_dir}")
        self.logger.info(f"{'='*60}")
    
    def save_evaluation_results(self, results_data: Dict) -> str:
        """Save evaluation results to JSON."""
        results_filename = self.results_dir / f"evaluation_{self.session_id}.json"
        
        with open(results_filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"📊 Evaluation results saved: {results_filename}")
        return str(results_filename)


class ModelSetupLogger:
    """Simple logging for model setup operations."""
    
    def __init__(self, model_name: str, timestamp: Optional[str] = None):
        self.model_name = model_name
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{model_name}_{self.timestamp}"
        
        # Setup directories
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        self.log_filename = self.log_dir / f"model_setup_{self.session_id}.log"
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers."""
        logger = logging.getLogger(f"model_setup_{self.session_id}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler only (keep it simple for model setup)
        console_handler = logging.StreamHandler()
        
        # Formatter
        formatter = logging.Formatter('%(message)s')  # Simple format for model setup
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger
    
    def log_model_creation_start(self, model_name: str, model_id: str):
        """Log model creation start."""
        self.logger.info(f"🤖 Creating {model_name} model...")
        self.logger.info(f"🏷️  Model identifier: {model_id}")
    
    def log_model_config(self, model_name: str, model_config: Dict):
        """Log model-specific configuration."""
        self.logger.info(f"📋 Using model-specific config for {model_name}")
        self.logger.info(f"   LR: {model_config.get('learning_rate', 'default')}")
        self.logger.info(f"   Batch: {model_config.get('batch_size', 'default')}")
        self.logger.info(f"   Dropout: {model_config.get('dropout', 'default')}")
    
    def log_dinov2_info(self, model_name: str, embed_dim: int, freeze_backbone: bool):
        """Log DINOv2 specific information."""
        self.logger.info(f"🦖 Loading DINOv2 model: {model_name}")
        self.logger.info(f"📐 DINOv2 embedding dimension: {embed_dim}")
        if freeze_backbone:
            self.logger.info("🧊 Backbone frozen (will unfreeze later if specified)")
        else:
            self.logger.info("🔥 Backbone unfrozen from start")
    
    def log_gradient_checkpointing(self, enabled: bool, success: bool = True, error: str = None):
        """Log gradient checkpointing status."""
        if enabled:
            if success:
                self.logger.info("✅ Gradient checkpointing enabled")
            else:
                self.logger.info(f"⚠️  Could not enable gradient checkpointing: {error}")
        else:
            self.logger.info("⚠️  Gradient checkpointing not supported by this model")
    
    def log_model_compilation(self, success: bool, error: str = None):
        """Log model compilation status."""
        if success:
            self.logger.info("🚀 Model compiled for faster training")
        elif error:
            self.logger.info(f"⚠️  Model compilation failed: {error}")
            self.logger.info("💡 Continuing without compilation...")
        else:
            self.logger.info("⚠️  torch.compile not available (requires PyTorch 2.0+)")
    
    def log_model_creation_complete(self, model_name: str, num_classes: int):
        """Log model creation completion."""
        self.logger.info(f"✅ {model_name} created with {num_classes} classes")
    
    def log_optimizer_creation(self, optimizer_name: str, params: Dict):
        """Log optimizer creation."""
        self.logger.info(f"🎯 Creating {optimizer_name.upper()} optimizer with params: {params}")
    
    def log_scheduler_creation(self, scheduler_name: str, params: Dict, use_warmup: bool = False, warmup_info: Dict = None):
        """Log scheduler creation."""
        self.logger.info(f"📅 Creating {scheduler_name.upper()} scheduler with params: {params}")
        if use_warmup and warmup_info:
            self.logger.info(f"🔥 Warmup enabled: {warmup_info['epochs']} epochs, {warmup_info['start_lr']} → {warmup_info['main_lr']}")
    
    def log_setup_complete(self, device: str, total_params: int, trainable_params: int, 
                          optimizer: str, scheduler: str, warmup_enabled: bool = False):
        """Log complete setup summary."""
        self.logger.info(f"\n=== MODEL SETUP COMPLETE ===")
        self.logger.info(f"📱 Device: {device}")
        self.logger.info(f"📊 Total parameters: {total_params:,}")
        self.logger.info(f"🎯 Trainable parameters: {trainable_params:,}")
        self.logger.info(f"⚖️  Class weights: Enabled")
        self.logger.info(f"🎯 Optimizer: {optimizer.upper()}")
        self.logger.info(f"📅 Scheduler: {scheduler.upper()}")
        if warmup_enabled:
            self.logger.info(f"🔥 Warmup: Enabled")
    
    def log_error(self, error_msg: str):
        """Log error messages."""
        self.logger.error(f"❌ {error_msg}")
    
    def log_warning(self, warning_msg: str):
        """Log warning messages."""
        self.logger.warning(f"⚠️  {warning_msg}")


class ComparisonLogger:
    """Centralized logging for model comparison experiments."""
    
    def __init__(self, timestamp: Optional[str] = None):
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"comparison_{self.timestamp}"
        
        # Setup directories
        self.log_dir = Path("logs")
        self.results_dir = Path("comparison_results")
        self.log_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        self.log_filename = self.log_dir / f"comparison_{self.session_id}.log"
        
        # Experiment tracking
        self.experiment_start = datetime.datetime.now()
        self.model_results = []
        self.experiment_metadata = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers."""
        logger = logging.getLogger(f"comparison_{self.session_id}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        log_file = self.log_dir / f"comparison_{self.session_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_experiment_start(self, dataset_info: Dict, experiment_config: Dict):
        """Log experiment start information."""
        self.experiment_metadata.update({
            'dataset_info': dataset_info,
            'experiment_config': experiment_config,
            'start_time': self.experiment_start.isoformat()
        })
        
        self.logger.info("🥊 STARTING MODEL COMPARISON EXPERIMENT")
        self.logger.info("=" * 60)
        self.logger.info(f"🔖 Session ID: {self.session_id}")
        self.logger.info(f"⏰ Started at: {self.experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"📄 Log file: {self.log_filename}")
        
        # Dataset info
        total_size = dataset_info.get('total_dataset_size', 0)
        sample_size = experiment_config.get('sample_size')
        if sample_size:
            percentage = (sample_size / total_size) * 100 if total_size > 0 else 0
            self.logger.info(f"📊 Dataset: {sample_size:,} samples ({percentage:.1f}% of {total_size:,})")
        else:
            self.logger.info(f"📊 Dataset: Full dataset ({total_size:,} samples)")
        
        # Few-shot info
        few_shot_mode = experiment_config.get('few_shot_mode')
        if few_shot_mode:
            few_shot_value = experiment_config.get('few_shot_value')
            self.logger.info(f"🎯 Few-shot: {few_shot_mode} ({few_shot_value}) - LABEL HIDING mode")
            self.logger.info(f"   💡 Models see ALL images but only some have labels!")
        else:
            self.logger.info(f"🎯 Few-shot: Disabled (all images have labels)")
        
        # Models to compare
        models = experiment_config.get('models', [])
        self.logger.info(f"🔧 Models to compare:")
        for model_type, actual_name in models:
            self.logger.info(f"   {model_type.upper()}: {actual_name}")
    
    def log_model_start(self, model_type: str, model_name: str, config_info: Dict):
        """Log individual model training start."""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"🧪 TESTING {model_type.upper()}: {model_name}")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"⚙️  Epochs: {config_info.get('epochs', 'N/A')}")
        self.logger.info(f"📋 Batch size: {config_info.get('batch_size', 'N/A')}")
        self.logger.info(f"📐 Image size: {config_info.get('image_size', 'N/A')}")
        self.logger.info(f"📈 Learning rate: {config_info.get('learning_rate', 'N/A')}")
        
        # Few-shot specific info
        few_shot_mode = config_info.get('few_shot_mode')
        if few_shot_mode:
            self.logger.info(f"🎯 Few-shot: {few_shot_mode} ({config_info.get('few_shot_value', 'N/A')})")
    
    def log_dataset_processing(self, model_type: str, original_sizes: Dict, final_sizes: Dict, few_shot_info: Dict = None):
        """Log dataset processing information."""
        self.logger.info(f"📊 Dataset processing for {model_type.upper()}:")
        self.logger.info(f"   Original: Train={original_sizes.get('train', 0):,}, "
                        f"Val={original_sizes.get('val', 0):,}, Test={original_sizes.get('test', 0):,}")
        self.logger.info(f"   Final: Train={final_sizes.get('train', 0):,}, "
                        f"Val={final_sizes.get('val', 0):,}, Test={final_sizes.get('test', 0):,}")
        
        if few_shot_info:
            labeled_samples = few_shot_info.get('labeled_samples', 0)
            total_samples = final_sizes.get('train', 0)
            if total_samples > 0:
                self.logger.info(f"   🏷️  Labeled samples: {labeled_samples:,}/{total_samples:,} "
                               f"({labeled_samples/total_samples*100:.1f}%)")
    
    def log_model_complete(self, model_type: str, model_name: str, result: Dict):
        """Log individual model completion."""
        if result.get('success', False):
            self.logger.info(f"\n✅ {model_type.upper()} TRAINING COMPLETE!")
            self.logger.info(f"   🎯 Train Accuracy: {result.get('train_accuracy', 0):.2f}%")
            self.logger.info(f"   🎯 Test Accuracy: {result.get('test_accuracy', 0):.2f}%")
            self.logger.info(f"   📈 Best Val: {result.get('best_val_acc', 0):.2f}%")
            self.logger.info(f"   ⏱️  Time: {result.get('time', 0):.1f}s")
            self.model_results.append(result)
        else:
            self.logger.error(f"\n❌ {model_type.upper()} TRAINING FAILED!")
            self.logger.error(f"   💥 Error: {result.get('error', 'Unknown error')}")
            self.logger.error(f"   ⏱️  Time: {result.get('time', 0):.1f}s")
    
    def log_gpu_cleanup(self):
        """Log GPU memory cleanup."""
        self.logger.info("🧹 Clearing GPU cache between models...")
    
    def log_comparison_results(self, results: List[Dict]):
        """Log final comparison results."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("📊 COMPARISON RESULTS")
        self.logger.info("=" * 60)
        
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        # Log successful results
        for result in successful_results:
            self.logger.info(f"\n🏆 {result['model_type'].upper()}: {result['model_name']}")
            self.logger.info(f"   🎯 Train Accuracy: {result['train_accuracy']:.2f}%")
            self.logger.info(f"   🎯 Test Accuracy: {result['test_accuracy']:.2f}%")
            self.logger.info(f"   📈 Best Val: {result['best_val_acc']:.2f}%")
            self.logger.info(f"   ⏱️  Time: {result['time']:.1f}s")
            
            # Dataset info
            train_samples = result.get('train_samples', 0)
            val_samples = result.get('val_samples', 0)
            test_samples = result.get('test_samples', 0)
            labeled_samples = result.get('labeled_samples', train_samples)
            
            if result.get('few_shot_mode'):
                self.logger.info(f"   📊 Samples: {train_samples} total train ({labeled_samples} labeled), "
                               f"{val_samples} val, {test_samples} test")
                self.logger.info(f"   🎯 Label ratio: {labeled_samples}/{train_samples} "
                               f"({labeled_samples/train_samples*100:.1f}%)")
            else:
                self.logger.info(f"   📊 Samples: {train_samples} train, {val_samples} val, {test_samples} test")
            
            self.logger.info(f"   ⚙️  Config: batch_size={result.get('batch_size', 'N/A')}, "
                           f"image_size={result.get('image_size', 'N/A')}, epochs={result.get('epochs', 'N/A')}")
            
            few_shot_info = 'Disabled' if result.get('few_shot_mode') is None else f"{result.get('few_shot_mode')} ({result.get('few_shot_value')})"
            self.logger.info(f"   🎯 Few-shot: {few_shot_info}")
        
        # Log failed results
        for result in failed_results:
            self.logger.error(f"\n💥 {result['model_type'].upper()}: FAILED")
            self.logger.error(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        # Determine winner
        if successful_results:
            best = max(successful_results, key=lambda x: x.get('test_accuracy', 0))
            self.logger.info(f"\n🏅 WINNER: {best['model_type'].upper()} ({best['test_accuracy']:.2f}%)")
            
            # Performance analysis
            if len(successful_results) >= 2:
                sorted_results = sorted(successful_results, key=lambda x: x.get('test_accuracy', 0), reverse=True)
                best_acc = sorted_results[0].get('test_accuracy', 0)
                second_acc = sorted_results[1].get('test_accuracy', 0)
                gap = best_acc - second_acc
                self.logger.info(f"📊 Performance gap: {gap:.2f}% advantage")
                
                if gap < 2:
                    self.logger.info("💡 Results are very close - consider statistical significance")
                elif gap > 10:
                    self.logger.info("💡 Significant performance difference detected")
    
    def log_experiment_summary(self):
        """Log experiment completion summary."""
        experiment_end = datetime.datetime.now()
        total_time = experiment_end - self.experiment_start
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🏁 EXPERIMENT COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"⏰ Total experiment time: {total_time}")
        self.logger.info(f"📊 Models tested: {len(self.model_results)}")
        self.logger.info(f"✅ Successful: {len([r for r in self.model_results if r.get('success', False)])}")
        self.logger.info(f"❌ Failed: {len([r for r in self.model_results if not r.get('success', False)])}")
        self.logger.info(f"📄 Log saved: {self.log_filename}")
    
    def save_comparison_results(self, results: List[Dict]) -> str:
        """Save comparison results to JSON file."""
        experiment_end = datetime.datetime.now()
        
        comparison_data = {
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'start_time': self.experiment_start.isoformat(),
            'end_time': experiment_end.isoformat(),
            'total_time': str(experiment_end - self.experiment_start),
            'experiment_metadata': self.experiment_metadata,
            'model_results': results,
            'summary': {
                'total_models': len(results),
                'successful_models': len([r for r in results if r.get('success', False)]),
                'failed_models': len([r for r in results if not r.get('success', False)]),
                'winner': max(results, key=lambda x: x.get('test_accuracy', 0)) if results else None
            },
            'files': {
                'log_file': str(self.log_filename)
            }
        }
        
        results_file = self.results_dir / f"comparison_{self.session_id}.json"
        with open(results_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        self.logger.info(f"📊 Comparison results saved: {results_file}")
        return str(results_file)


class DataSplitterLogger:
    """Simple logging for data splitting operations."""
    
    def __init__(self, dataset_name: str = "dataset", timestamp: Optional[str] = None):
        self.dataset_name = dataset_name
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{dataset_name}_{self.timestamp}"
        
        # Setup directories
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        self.log_filename = self.log_dir / f"data_split_{self.session_id}.log"
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers."""
        logger = logging.getLogger(f"data_split_{self.session_id}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler only (keep it simple for data splitting)
        console_handler = logging.StreamHandler()
        
        # Formatter
        formatter = logging.Formatter('%(message)s')  # Simple format for data splitting
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger
    
    def log_dataset_loading(self, pickle_path: str, dataset_format: str, total_images: int, num_classes: int = None, corrupted_count: int = 0):
        """Log dataset loading information."""
        self.logger.info("📂 Loading clean dataset...")
        self.logger.info(f"📋 Loaded {dataset_format} format dataset")
        if num_classes:
            self.logger.info(f"📋 Loaded dataset with {total_images:,} images, {num_classes} classes")
            self.logger.info(f"📊 Removed {corrupted_count} corrupted images during cleaning")
    
    def log_path_conversion(self, base_data_dir: str, conversion_type: str = "relative"):
        """Log path conversion operations."""
        if conversion_type == "old_to_new":
            self.logger.info(f"🔗 Converting old paths to new base: {base_data_dir}")
        else:
            self.logger.info(f"🔗 Converting relative paths using base: {base_data_dir}")
    
    def log_split_configuration(self, train_size: float, val_size: float, test_size: float, total_images: int):
        """Log split configuration."""
        self.logger.info(f"📁 Processing {total_images:,} clean images")
        self.logger.info(f"📊 Target split: Train {train_size:.1%}, Val {val_size:.1%}, Test {test_size:.1%}")
    
    def log_stratification_warning(self, min_class_count: int):
        """Log stratification warnings."""
        self.logger.warning(f"⚠️  WARNING: Some classes have very few samples (min: {min_class_count})")
        self.logger.warning("    This may cause stratification issues.")
    
    def log_split_summary(self, train_count: int, val_count: int, test_count: int, total_images: int):
        """Log split summary."""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"📊 DATA SPLIT SUMMARY")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"🖼️  Total images: {total_images:,}")
        self.logger.info(f"🏷️  Total labels: {total_images:,} (All images have labels at this stage)")
        self.logger.info(f"Training:   {train_count:,} ({train_count/total_images*100:.1f}%)")
        self.logger.info(f"Validation: {val_count:,} ({val_count/total_images*100:.1f}%)")
        self.logger.info(f"Test:       {test_count:,} ({test_count/total_images*100:.1f}%)")
    
    def log_class_distribution_verification(self, total_classes: int, train_classes: int, val_classes: int, test_classes: int,
                                          missing_train: set, missing_val: set, missing_test: set):
        """Log class distribution verification."""
        self.logger.info(f"\n📋 Class distribution verification:")
        self.logger.info(f"Total classes: {total_classes}")
        self.logger.info(f"Train classes: {train_classes}")
        self.logger.info(f"Val classes:   {val_classes}")
        self.logger.info(f"Test classes:  {test_classes}")
        
        if missing_train or missing_val or missing_test:
            self.logger.warning(f"\n⚠️  WARNING: Some classes missing from splits!")
            if missing_train:
                self.logger.warning(f"   Missing from train: {missing_train}")
            if missing_val:
                self.logger.warning(f"   Missing from val: {missing_val}")
            if missing_test:
                self.logger.warning(f"   Missing from test: {missing_test}")
        else:
            self.logger.info(f"✅ All classes present in all splits!")
    
    def log_per_class_distribution(self, all_classes: List[str], train_dist: Dict, val_dist: Dict, test_dist: Dict, show_limit: int = 5):
        """Log per-class distribution details."""
        self.logger.info(f"\n📈 Per-class split verification (first {show_limit} classes):")
        
        for i, class_name in enumerate(sorted(all_classes)):
            if i >= show_limit:
                break
            train_count = train_dist.get(class_name, 0)
            val_count = val_dist.get(class_name, 0)
            test_count = test_dist.get(class_name, 0)
            total_count = train_count + val_count + test_count
            
            self.logger.info(f"  {class_name}:")
            self.logger.info(f"    Train: {train_count:,} ({train_count/total_count*100:.1f}%)")
            self.logger.info(f"    Val:   {val_count:,} ({val_count/total_count*100:.1f}%)")
            self.logger.info(f"    Test:  {test_count:,} ({test_count/total_count*100:.1f}%)")
        
        if len(all_classes) > show_limit:
            self.logger.info(f"  ... and {len(all_classes) - show_limit} more classes")
    
    def log_data_inspection(self, train_paths: List[str], train_labels: List[str], val_paths: List[str], test_paths: List[str]):
        """Log quick data inspection."""
        total_classes = len(set(train_labels + []))  # Simplified for train labels only
        self.logger.info(f"📊 Total: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
        self.logger.info(f"📁 Sample paths:")
        for i in range(min(2, len(train_paths))):
            path_exists = os.path.exists(train_paths[i]) if train_paths else False
            self.logger.info(f"   {train_labels[i] if train_labels else 'N/A'}: {train_paths[i] if train_paths else 'N/A'} - Exists: {path_exists}")
        self.logger.info(f"🏷️  Classes: {total_classes} total")
        self.logger.info(f"\n✅ Dataset split completed successfully!")
    
    def log_few_shot_disabled(self, total_count: int):
        """Log few-shot learning disabled."""
        self.logger.info(f"\n📊 FEW-SHOT LEARNING: DISABLED")
        self.logger.info(f"   🖼️  Using all available training data: {total_count} samples")
        self.logger.info(f"   🖼️  Total training images: {total_count}")
        self.logger.info(f"   🏷️  Images with labels: {total_count} (100%)")
        self.logger.info(f"   ❓ Images without labels: 0 (0%)")
    
    def log_few_shot_enabled(self, mode: str, value: float):
        """Log few-shot learning enabled."""
        self.logger.info(f"\n📊 FEW-SHOT LEARNING: ACTIVATED")
        self.logger.info(f"   🎯 Mode: {mode} = {value}")
        if mode == 'percentage':
            self.logger.info(f"   💡 Using {value*100:.1f}% of labeled training data")
        elif mode == 'per_class':
            self.logger.info(f"   💡 Using {value} labeled samples per class")
    
    def log_few_shot_results(self, original_count: int, labeled_count: int, class_counts: Dict):
        """Log few-shot learning results."""
        unlabeled_count = original_count - labeled_count
        self.logger.info(f"📊 FEW-SHOT SIMULATION RESULTS:")
        self.logger.info(f"   🖼️  Total images available: {original_count}")
        self.logger.info(f"   🏷️  Images with labels: {labeled_count} ({labeled_count/original_count*100:.1f}%)")
        self.logger.info(f"   ❓ Images without labels: {unlabeled_count} ({unlabeled_count/original_count*100:.1f}%)")
        self.logger.info(f"   💡 This simulates real-world scenario: abundant images, scarce labels")
        self.logger.info(f"📈 Class distribution of LABELED data:")
        for class_name, count in sorted(class_counts.items()):
            self.logger.info(f"   {class_name}: {count} labeled samples")
    
    def log_path_conversion_warning(self, old_path: str, base_dir_name: str):
        """Log path conversion warnings."""
        self.logger.warning(f"⚠️  Base directory '{base_dir_name}' not found in path: {old_path}")
        self.logger.warning("    Using original path - this may cause file not found errors")
    
    def log_path_conversion_error(self, old_path: str, error: str):
        """Log path conversion errors."""
        self.logger.error(f"❌ Error parsing path '{old_path}': {error}")
        self.logger.error("    Using original path as fallback")