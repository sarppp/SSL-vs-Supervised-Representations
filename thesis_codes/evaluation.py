import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast
import json
import os
from datetime import datetime
from logger_manager import EvaluationLogger

def evaluate_model(model, test_loader, device, class_names, use_amp=False, eval_logger=None):
    """Evaluate model on test set - only use this AFTER training is complete!"""
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            if use_amp:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    
    # Calculate overall accuracy
    test_accuracy = accuracy_score(y_true, y_pred) * 100
    
    if eval_logger:
        eval_logger.log_test_accuracy(test_accuracy)
    
    return y_true, y_pred, y_prob, test_accuracy

def analyze_class_balance_performance(y_true, y_pred, class_names, eval_logger=None):
    """Analyze if class weights worked - check for prediction imbalance."""
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    if eval_logger:
        eval_logger.log_class_balance_analysis(class_names, true_counts, pred_counts)
    
    return true_counts, pred_counts

def detailed_classification_report(y_true, y_pred, class_names, eval_logger=None):
    """Generate detailed per-class performance metrics with over/underfitting warnings."""
    if eval_logger:
        eval_logger.log_per_class_accuracy(class_names, y_true, y_pred)

def comprehensive_test_evaluation(model, test_loader, device, class_names, model_name=None, config_module=None, use_amp=False):
    """Complete test evaluation pipeline (without confusion matrix)."""
    # Use default config if none provided (for backward compatibility)
    if config_module is None:
        import config
        config_module = config
    
    model_name = model_name or config_module.MODEL_NAME
    
    # Create evaluation logger
    eval_logger = EvaluationLogger(model_name)
    
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
        model_name, test_accuracy, len(y_true), class_names, 
        few_shot_enabled, config_dict, getattr(config_module, 'SAVE_DIR', None)
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
    
    evaluation_results = {
        'session_id': eval_logger.session_id,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'test_accuracy': float(test_accuracy),
        'total_test_samples': int(len(y_true)),
        'num_classes': len(class_names),
        'class_names': class_names,
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
        'per_class_accuracy': per_class_acc
    }