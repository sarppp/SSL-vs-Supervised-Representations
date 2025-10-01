import torch
import numpy as np
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score
)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp.autocast_mode import autocast
import json
import os
from datetime import datetime
from src.utils.logger_manager import EvaluationLogger
from src.config import config_paths
from src.models.model_setup import get_model_identifier

def evaluate_model(model, test_loader, device, class_names, use_amp=False, eval_logger=None):
    """Evaluate model on test set - only use this AFTER training is complete!"""
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    # 🔧 FIX: Process in smaller chunks to reduce memory pressure
    print(f"🔍 Evaluating {len(test_loader.dataset)} samples in {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # 🔧 Add progress info for large test sets
            if batch_idx % 20 == 0:
                print(f"   Processing batch {batch_idx+1}/{len(test_loader)}...")
                
            images, labels = images.to(device), labels.to(device)
            
            if use_amp:
                with autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # 🔧 FIX: Clear GPU cache periodically during large evaluations
            if batch_idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    
    # Calculate overall accuracy
    test_accuracy = accuracy_score(y_true, y_pred) * 100
    
    if eval_logger:
        eval_logger.log_test_accuracy(test_accuracy)
    
    print(f"✅ Evaluation complete: {test_accuracy:.2f}% accuracy on {len(y_true)} samples")
    
    return y_true, y_pred, y_prob, test_accuracy

def analyze_class_balance_performance(y_true, y_pred, class_names, eval_logger=None):
    """Analyze if class weights worked - check for prediction imbalance."""
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    if eval_logger:
        eval_logger.log_class_balance_analysis(class_names, true_counts, pred_counts)
    
    return true_counts, pred_counts

def save_confusion_matrix_plot(cm, class_names, model_name, config_module=None):
    """
    🎓 THESIS: Save confusion matrix as high-quality plot for thesis/papers
    """
    from src.config import config_paths
    
    # Create output directory for plots
    plots_dir = os.path.join(config_paths.OUTPUTS_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Derive label-budget suffix based on config_module if available
    budget_suffix = ""
    try:
        if config_module is not None:
            mode = getattr(config_module, 'FEW_SHOT_MODE', None)
            value = getattr(config_module, 'FEW_SHOT_VALUE', None)
            if mode == 'percentage' and value is not None:
                budget_suffix = f"_labels{int(float(value)*100)}pct"
            elif mode == 'per_class' and value is not None:
                budget_suffix = f"_labels{int(value)}perclass"
            elif mode is None:
                budget_suffix = "_labels100pct"
    except Exception:
        budget_suffix = ""

    # Include training regime in filename if known
    regime_suffix = ""
    try:
        if config_module is not None:
            regime = getattr(config_module, 'TRAINING_REGIME', None)
            if regime:
                regime_suffix = f"_{str(regime)}"
    except Exception:
        regime_suffix = ""

    base_filename = f'confusion_matrix_{model_name}{regime_suffix}{budget_suffix}'
    png_path = os.path.join(plots_dir, f'{base_filename}.png')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Confusion matrix saved:")
    print(f"   PNG: {png_path}")
    
    return png_path

def detailed_classification_report(y_true, y_pred, class_names, eval_logger=None):
    """Generate detailed per-class performance metrics with over/underfitting warnings."""
    if eval_logger:
        eval_logger.log_per_class_accuracy(class_names, y_true, y_pred)

def comprehensive_test_evaluation(model, test_loader, device, class_names, model_name=None, config_module=None, use_amp=False, eval_logger=None):
    """Complete test evaluation pipeline (without confusion matrix)."""
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        from src.config import config
        config_module = config
    
    model_name = model_name or config_module.MODEL_NAME
    
    # Generate full model identifier
    model_identifier = get_model_identifier(model_name, config_module)
    
    # Create evaluation logger only if not provided
    if eval_logger is None:
        eval_logger = EvaluationLogger(model_identifier, model_name=model_name)
        is_shared_logger = False
    else:
        is_shared_logger = True
    
    # Check if this was trained with few-shot learning
    few_shot_enabled = hasattr(config_module, 'FEW_SHOT_MODE') and config_module.FEW_SHOT_MODE is not None
    
    # Prepare config dict for logging
    config_dict = {k: v for k, v in config_module.__dict__.items() 
                   if not k.startswith('__') and not callable(v)}
    
    # Log evaluation start
    eval_logger.log_evaluation_start(few_shot_enabled, config_dict)
    
    # 1. Evaluate model
    y_true, y_pred, y_prob, test_accuracy = evaluate_model(model, test_loader, device, class_names, use_amp, eval_logger)
    
    # 2. Analyze class balance performance
    true_counts, pred_counts = analyze_class_balance_performance(y_true, y_pred, class_names, eval_logger)
    
    # 3. Detailed classification report (with warnings)
    detailed_classification_report(y_true, y_pred, class_names, eval_logger)
    
    # 4. Summary
    eval_logger.log_evaluation_summary(
        model_name, float(test_accuracy), len(y_true), class_names, 
        few_shot_enabled, config_dict, config_paths.MODELS_DIR
    )
    
    # Calculate per-class accuracies
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        class_count = class_mask.sum()
        if class_count > 0:
            class_acc = (y_pred[class_mask] == i).mean() * 100
            per_class_acc[class_name] = {
                'accuracy': float(class_acc),
                'sample_count': int(class_count)
            }
    
    # 🎓 THESIS-QUALITY METRICS: Precision, Recall, F1, Confusion Matrix, etc.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and weighted averages (essential for thesis)
    precision_macro = float(precision.mean())
    recall_macro = float(recall.mean())
    f1_macro = float(f1.mean())
    
    precision_weighted = float(np.average(precision, weights=support))
    recall_weighted = float(np.average(recall, weights=support))
    f1_weighted = float(np.average(f1, weights=support))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Matthews Correlation Coefficient (robust metric)
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    # Multi-class ROC-AUC (if probabilities available)
    try:
        # Check if y_prob is valid and has the right shape
        if y_prob is not None and len(y_prob) > 0 and y_prob.shape[1] > 1:
            # Ensure probabilities are valid (not NaN, not all zeros)
            if not np.any(np.isnan(y_prob)) and not np.all(y_prob == 0):
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
            else:
                print(f"⚠️ Invalid probabilities for ROC-AUC: NaN={np.any(np.isnan(y_prob))}, All zeros={np.all(y_prob == 0)}")
                roc_auc = None
        else:
            print(f"⚠️ Invalid y_prob shape for ROC-AUC: {y_prob.shape if y_prob is not None else 'None'}")
            roc_auc = None
    except Exception as e:
        print(f"⚠️ ROC-AUC calculation failed: {e}")
        roc_auc = None
    
    # Per-class detailed metrics (for thesis tables)
    per_class_detailed = {}
    for i, class_name in enumerate(class_names):
        per_class_detailed[class_name] = {
            'accuracy': per_class_acc[class_name]['accuracy'],
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    evaluation_results = {
        'session_id': eval_logger.session_id,
        'model_name': model_name,
        'test_accuracy': float(test_accuracy),
        'total_test_samples': int(len(y_true)),
        'num_classes': len(class_names),
        'class_names': class_names,
        
        # 🎓 THESIS METRICS: Overall scores
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'matthews_corrcoef': mcc,
        'roc_auc_ovr': roc_auc,
        
        # 🎓 THESIS METRICS: Per-class detailed
        'per_class_metrics': per_class_detailed,
        
        # 🎓 THESIS METRICS: Confusion matrix
        'confusion_matrix': cm.tolist(),
        
        # Original metrics
        'few_shot_enabled': few_shot_enabled,
        'few_shot_mode': config_module.FEW_SHOT_MODE if few_shot_enabled else None,
        'few_shot_value': config_module.FEW_SHOT_VALUE if few_shot_enabled else None,
        'per_class_accuracy': per_class_acc,
        'true_distribution': {class_names[i]: int(count) for i, count in true_counts.items()},
        'predicted_distribution': {class_names[i]: int(count) for i, count in pred_counts.items()},
        'predictions': y_pred.tolist(),
        'true_labels': y_true.tolist(),
        'probabilities': y_prob.tolist(),
        'files': {
            'log_file': str(eval_logger.log_filename),
            'results_file': None  # Will be set by save_evaluation_results
        }
    }
    
    # 🎓 THESIS: Save confusion matrix visualization
    try:
        save_confusion_matrix_plot(cm, class_names, model_name, config_module)
    except Exception as e:
        print(f"⚠️ Could not save confusion matrix plot: {e}")
    
    # Save results using evaluation logger
    results_filename = eval_logger.save_evaluation_results(evaluation_results)
    
    return {
        'test_accuracy': test_accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'true_counts': true_counts,
        'pred_counts': pred_counts,
        'class_names': class_names,
        'log_file': str(eval_logger.log_filename),
        'results_file': results_filename,
        'per_class_accuracy': per_class_acc,
        # 🎓 THESIS: Include comprehensive metrics
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    # Minimal informative print when run as a module script.
    try:
        logs_dir = config_paths.LOGS_DIR
        eval_dir = config_paths.EVALUATION_RESULTS_DIR
        outputs_dir = config_paths.OUTPUTS_DIR
        print("Evaluation module loaded. This module exposes functions for programmatic evaluation.")
        print(f"Logs directory: {logs_dir}")
        print(f"Evaluation results directory: {eval_dir}")
        print(f"Outputs root: {outputs_dir}")

        # Show latest evaluation JSON if exists
        import glob, os
        pattern = os.path.join(eval_dir, "evaluation_*.json")
        files = glob.glob(pattern)
        if files:
            latest = max(files, key=os.path.getmtime)
            print(f"Latest evaluation results: {latest}")
        else:
            print("No evaluation results found yet. Run your training/evaluation pipeline first.")
    except Exception as e:
        print(f"Could not display evaluation info: {e}")