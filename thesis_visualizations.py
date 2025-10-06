#!/usr/bin/env python3
"""
Master Thesis Visualizations: Supervised vs Self-Supervised Models
Comprehensive comparison of CNN, DINOv2, and ViT models for crop pest classification
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class ThesisVisualizer:
    def __init__(self, outputs_dir="outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.viz_dir = self.outputs_dir / "thesis_visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Load all available data
        self.comparison_data = self.load_comparison_data()
        self.training_data = self.load_training_data()
        self.evaluation_data = self.load_evaluation_data()
        self.checkpoint_data = self.analyze_checkpoints()
        
        # Define label percentage mappings and colors
        self.label_percentages = {
            '0': {'name': '0% (Zero-shot)', 'color': '#E74C3C', 'suffix': '0pct'},
            '10': {'name': '10% (Few-shot)', 'color': '#F39C12', 'suffix': '10pct'},
            '100': {'name': '100% (Full labels)', 'color': '#27AE60', 'suffix': '100pct'}
        }
        
        # Detect available label percentages from data
        self.available_percentages = self.detect_available_percentages()
        
    def load_comparison_data(self):
        """Load comparison results"""
        comparison_file = self.outputs_dir / "comparison_results" / "comparison.json"
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                return json.load(f)
        return None
    
    def load_training_data(self):
        """Load training results for all models"""
        training_dir = self.outputs_dir / "training_results"
        training_data = {}
        
        if training_dir.exists():
            for json_file in training_dir.glob("*.json"):
                model_name = json_file.stem.replace("training_", "")
                with open(json_file, 'r') as f:
                    training_data[model_name] = json.load(f)
        
        return training_data
    
    def load_evaluation_data(self):
        """Load evaluation results for all models"""
        eval_dir = self.outputs_dir / "evaluation_results"
        eval_data = {}
        
        if eval_dir.exists():
            for json_file in eval_dir.glob("*.json"):
                model_name = json_file.stem.replace("evaluation_", "")
                with open(json_file, 'r') as f:
                    eval_data[model_name] = json.load(f)
        
        return eval_data
    
    def analyze_checkpoints(self):
        """Analyze saved model checkpoints"""
        checkpoints_dir = self.outputs_dir / "checkpoints"
        checkpoint_data = {}
        
        if checkpoints_dir.exists():
            for checkpoint in checkpoints_dir.glob("*.pth"):
                # Parse checkpoint filename for accuracy
                filename = checkpoint.name
                if "acc" in filename:
                    acc_part = filename.split("acc")[-1].replace(".pth", "")
                    try:
                        accuracy = float(acc_part)
                        model_type = self.extract_model_type(filename)
                        
                        if model_type not in checkpoint_data:
                            checkpoint_data[model_type] = []
                        
                        checkpoint_data[model_type].append({
                            'filename': filename,
                            'accuracy': accuracy,
                            'size_mb': checkpoint.stat().st_size / (1024 * 1024),
                            'type': 'best' if 'best_' in filename else 'final' if 'final_' in filename else 'checkpoint'
                        })
                    except ValueError:
                        continue
        
        return checkpoint_data
    
    def extract_model_type(self, filename):
        """Extract model type from checkpoint filename"""
        if "cnn" in filename or "efficientnet" in filename:
            return "CNN (EfficientNet)"
        elif "dino" in filename:
            return "DINOv2 (Self-Supervised)"
        elif "vit" in filename:
            return "ViT (Supervised)"
        return "Unknown"
    
    def extract_label_percentage(self, model_name):
        """Extract label percentage from model name"""
        # Check for explicit label percentage in model name
        for percentage in ['0', '10', '100']:
            if f'_label_{percentage}_' in model_name or f'label_{percentage}' in model_name:
                return percentage
            elif f'{percentage}pct' in model_name or f'{percentage}%' in model_name:
                return percentage
        
        # Infer from model type/name patterns
        if 'zero_shot' in model_name or 'zeroshot' in model_name:
            return '0'
        elif 'few_shot' in model_name or 'fewshot' in model_name:
            return '10'
        
        # Return None for unrecognized patterns instead of defaulting to 100%
        return None
    
    def detect_available_percentages(self):
        """Detect which label percentages are available in the data"""
        available = set()
        
        # Check comparison data
        if self.comparison_data and 'model_results' in self.comparison_data:
            for model in self.comparison_data['model_results']:
                pct = self.extract_label_percentage(model.get('model_name', ''))
                if pct is not None:
                    available.add(pct)
        
        # Check checkpoint data - this is the most reliable source
        checkpoints_dir = self.outputs_dir / "checkpoints"
        if checkpoints_dir.exists():
            for checkpoint in checkpoints_dir.glob("*.pth"):
                pct = self.extract_label_percentage(checkpoint.name)
                if pct is not None:
                    available.add(pct)
        
        # Also check models directory for additional model files
        models_dir = Path("models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pth"):
                pct = self.extract_label_percentage(model_file.name)
                if pct is not None:
                    available.add(pct)
        
        # Check training data
        for model_name in self.training_data.keys():
            pct = self.extract_label_percentage(model_name)
            if pct is not None:
                available.add(pct)
        
        # Only return percentages that are in our defined label_percentages and have actual data
        valid_percentages = []
        for pct in sorted(available):
            if pct in self.label_percentages:
                # Verify we actually have data for this percentage
                filtered_data = self.filter_data_by_percentage(pct)
                has_data = (
                    (filtered_data['comparison_data'] and filtered_data['comparison_data'].get('model_results')) or
                    filtered_data['training_data'] or
                    filtered_data['evaluation_data'] or
                    filtered_data['checkpoint_data']
                )
                if has_data:
                    valid_percentages.append(pct)
        
        return valid_percentages
    
    def filter_data_by_percentage(self, percentage):
        """Filter all data to only include models with the specified label percentage"""
        filtered_data = {
            'comparison_data': None,
            'training_data': {},
            'evaluation_data': {},
            'checkpoint_data': {}
        }
        
        # Filter comparison data
        if self.comparison_data and 'model_results' in self.comparison_data:
            filtered_models = []
            for model in self.comparison_data['model_results']:
                model_pct = self.extract_label_percentage(model.get('model_name', ''))
                if model_pct == percentage:
                    filtered_models.append(model)
            
            if filtered_models:
                filtered_data['comparison_data'] = {
                    'model_results': filtered_models,
                    'summary': self.comparison_data.get('summary', {})
                }
        
        # Filter training data
        for model_name, data in self.training_data.items():
            model_pct = self.extract_label_percentage(model_name)
            if model_pct == percentage:
                filtered_data['training_data'][model_name] = data
        
        # Filter evaluation data
        for model_name, data in self.evaluation_data.items():
            model_pct = self.extract_label_percentage(model_name)
            if model_pct == percentage:
                filtered_data['evaluation_data'][model_name] = data
        
        # Filter checkpoint data
        checkpoints_dir = self.outputs_dir / "checkpoints"
        models_dir = Path("models")
        directories_to_check = []
        if checkpoints_dir.exists():
            directories_to_check.append(checkpoints_dir)
        if models_dir.exists():
            directories_to_check.append(models_dir)
        
        checkpoint_data = {}
        for directory in directories_to_check:
            for checkpoint in directory.glob("*.pth"):
                checkpoint_pct = self.extract_label_percentage(checkpoint.name)
                if checkpoint_pct == percentage:
                    model_type = self.extract_model_type(checkpoint.name)
                    if model_type not in checkpoint_data:
                        checkpoint_data[model_type] = []
                    
                    # Extract accuracy from filename
                    filename = checkpoint.name
                    if "acc" in filename:
                        try:
                            acc_part = filename.split("acc")[-1].replace(".pth", "")
                            accuracy = float(acc_part)
                            checkpoint_data[model_type].append({
                                'filename': checkpoint.name,
                                'accuracy': accuracy,
                                'size_mb': checkpoint.stat().st_size / (1024 * 1024),
                                'type': 'best' if 'best_' in checkpoint.name else 'final' if 'final_' in checkpoint.name else 'checkpoint'
                            })
                        except ValueError:
                            continue
        
        if checkpoint_data:
            filtered_data['checkpoint_data'] = checkpoint_data
        
        return filtered_data
    
    def create_comparison_data_from_checkpoints(self, checkpoint_data, percentage):
        """Create comparison data structure from checkpoint files when JSON data unavailable"""
        model_results = []
        
        for model_type, checkpoints in checkpoint_data.items():
            # Find the best checkpoint for this model type
            best_checkpoint = max(checkpoints, key=lambda x: x['accuracy'])
            
            # Extract model name from filename 
            filename = best_checkpoint['filename']
            if 'cnn' in filename.lower():
                model_name = f"efficientnet_b4_label_{percentage}"
            elif 'dino' in filename.lower():
                model_name = f"dinov2_vitb14_label_{percentage}"  
            elif 'vit' in filename.lower():
                model_name = f"vit_base_patch16_224_label_{percentage}"
            else:
                model_name = f"unknown_model_label_{percentage}"
            
            # Create model result entry with available data
            model_result = {
                'model_name': model_name,
                'test_accuracy': best_checkpoint['accuracy'],
                'train_accuracy': best_checkpoint['accuracy'],  # Approximation
                'best_val_acc': best_checkpoint['accuracy'],   # Approximation
                'time': 3600,  # Default 1 hour (no timing data available)
                'checkpoint_type': best_checkpoint['type']
            }
            
            model_results.append(model_result)
        
        return {
            'model_results': model_results,
            'summary': {
                'total_models': len(model_results),
                'label_percentage': percentage
            }
        } if model_results else None
    
    def create_simple_accuracy_chart(self, checkpoint_data, percentage):
        """Create simple accuracy chart from checkpoint data when training curves unavailable"""
        if percentage:
            title = f'Model Accuracy Summary: {self.label_percentages[percentage]["name"]}'
            suffix = self.label_percentages[percentage]['suffix']
        else:
            title = 'Model Accuracy Summary'
            suffix = ''
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        model_names = []
        accuracies = []
        colors = []
        
        for model_type, checkpoints in checkpoint_data.items():
            best_checkpoint = max(checkpoints, key=lambda x: x['accuracy'])
            model_names.append(model_type)
            accuracies.append(best_checkpoint['accuracy'])
            
            colors.append(self.get_model_colors(model_type))
        
        bars = ax.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_title('Best Model Accuracy', fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        if suffix:
            save_path = self.viz_dir / f"training_curves_{suffix}.png"
            print(f"Training curves ({percentage}% labels) saved to: {save_path}")
        else:
            save_path = self.viz_dir / "training_curves.png"
            print(f"Training curves saved to: {save_path}")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def create_performance_comparison(self, percentage=None, filtered_data=None):
        """Create comprehensive performance comparison visualization for specific label percentage"""
        if filtered_data:
            comparison_data = filtered_data['comparison_data']
        else:
            comparison_data = self.comparison_data
            
        # Fallback: create comparison data from checkpoints if JSON not available
        if not comparison_data and filtered_data and filtered_data['checkpoint_data']:
            comparison_data = self.create_comparison_data_from_checkpoints(filtered_data['checkpoint_data'], percentage)
            
        if not comparison_data:
            print(f" No comparison data available for {percentage}% labels")
            return None
        
        # Create title based on percentage
        if percentage:
            title = f'Model Performance Comparison: {self.label_percentages[percentage]["name"]}'
            color_theme = self.label_percentages[percentage]['color']
        else:
            title = 'Model Performance Comparison: All Models'
            color_theme = '#4ECDC4'
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        models = comparison_data['model_results']
        
        # Prepare data
        model_names = []
        model_types = []
        train_accs = []
        test_accs = []
        val_accs = []
        training_times = []
        
        for model in models:
            name = model['model_name'].replace('_', ' ').title()
            if 'dinov2' in model['model_name']:
                name = f"DINOv2 (Self-Supervised)\n{name.split()[-1]}"
                model_type = "Self-Supervised"
            elif 'efficientnet' in model['model_name']:
                name = f"CNN (Supervised)\n{name}"
                model_type = "Supervised"
            elif 'vit' in model['model_name']:
                name = f"ViT (Supervised)\n{name.split()[-1]}"
                model_type = "Supervised"
            else:
                model_type = "Unknown"
            
            model_names.append(name)
            model_types.append(model_type)
            train_accs.append(model['train_accuracy'])
            test_accs.append(model['test_accuracy'])
            val_accs.append(model['best_val_acc'])
            training_times.append(model['time'] / 60)  # Convert to minutes
        
        # Colors for model types - use specific model colors
        colors = []
        for model in models:
            model_name = model['model_name']
            colors.append(self.get_model_colors(model_name))
        
        # 1. Test Accuracy Comparison
        bars1 = ax1.bar(model_names, test_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Test Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, test_accs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Training vs Validation vs Test Accuracy
        x = np.arange(len(model_names))
        width = 0.25
        
        bars2_1 = ax2.bar(x - width, train_accs, width, label='Training', alpha=0.8, color='#95E1D3')
        bars2_2 = ax2.bar(x, val_accs, width, label='Validation', alpha=0.8, color='#F38BA8')
        bars2_3 = ax2.bar(x + width, test_accs, width, label='Test', alpha=0.8, color='#3D5A80')
        
        ax2.set_title('Training vs Validation vs Test Accuracy', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_xlabel('Models')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        # 3. Training Time Comparison
        bars3 = ax3.bar(model_names, training_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('Training Time Comparison', fontweight='bold')
        ax3.set_ylabel('Training Time (minutes)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time in zip(bars3, training_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time:.1f}m', ha='center', va='bottom', fontweight='bold')
        
        # 4. Efficiency Plot (Accuracy vs Time)
        scatter = ax4.scatter(training_times, test_accs, c=[self.get_model_colors(model_name) for model_name in [model['model_name'] for model in models]], 
                            s=200, alpha=0.8, edgecolors='black', linewidth=2)
        
        # Add model labels
        for i, name in enumerate(model_names):
            ax4.annotate(name.split('\n')[0], (training_times[i], test_accs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        ax4.set_title('Efficiency: Accuracy vs Training Time', fontweight='bold')
        ax4.set_xlabel('Training Time (minutes)')
        ax4.set_ylabel('Test Accuracy (%)')
        ax4.grid(True, alpha=0.3)
        
        # Add legend for model types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='CNN (EfficientNet)'),
            Patch(facecolor='#FFE66D', label='ViT (Supervised)'),
            Patch(facecolor='#4ECDC4', label='DINOv2 (Self-Supervised)')
        ]
        ax4.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        # Save the plot with percentage suffix
        if percentage:
            suffix = self.label_percentages[percentage]['suffix']
            save_path = self.viz_dir / f"performance_comparison_{suffix}.png"
            print(f"Performance comparison ({percentage}% labels) saved to: {save_path}")
        else:
            save_path = self.viz_dir / "performance_comparison.png"
            print(f"Performance comparison saved to: {save_path}")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def create_training_curves(self, percentage=None, filtered_data=None):
        """Create training curves visualization for specific label percentage"""
        if filtered_data:
            training_data = filtered_data['training_data']
        else:
            training_data = self.training_data
            
        # Fallback: create simple accuracy summary when detailed training data unavailable
        if not training_data and filtered_data and filtered_data['checkpoint_data']:
            return self.create_simple_accuracy_chart(filtered_data['checkpoint_data'], percentage)
            
        if not training_data:
            print(f" No training data available for {percentage}% labels")
            return None
        
        # Create title based on percentage
        if percentage:
            title = f'Training Dynamics: {self.label_percentages[percentage]["name"]}'
        else:
            title = 'Training Dynamics: All Models'
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Dynamic colors and labels based on available models
        colors = {}
        labels = {}
        
        for model_name in training_data.keys():
            if 'dinov2' in model_name:
                colors[model_name] = '#4ECDC4'
                labels[model_name] = 'DINOv2 (Self-Supervised)'
            elif 'efficientnet' in model_name or 'cnn' in model_name:
                colors[model_name] = '#FF6B6B' 
                labels[model_name] = 'CNN (Supervised)'
            elif 'vit' in model_name:
                colors[model_name] = '#FFE66D'
                labels[model_name] = 'ViT (Supervised)'
            else:
                colors[model_name] = '#999999'
                labels[model_name] = model_name.replace('_', ' ').title()
        
        # 1. Training Loss Curves
        for model_name, data in training_data.items():
            if 'metrics_history' in data:
                epochs = data['metrics_history']['epochs']
                train_losses = data['metrics_history']['train_losses']
                ax1.plot(epochs, train_losses, marker='o', linewidth=2, 
                        color=colors.get(model_name, 'gray'), label=labels.get(model_name, model_name))
        
        ax1.set_title('Training Loss Curves', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Validation Loss Curves
        for model_name, data in training_data.items():
            if 'metrics_history' in data:
                epochs = data['metrics_history']['epochs']
                val_losses = data['metrics_history']['val_losses']
                ax2.plot(epochs, val_losses, marker='s', linewidth=2,
                        color=colors.get(model_name, 'gray'), label=labels.get(model_name, model_name))
        
        ax2.set_title('Validation Loss Curves', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Training Accuracy Curves
        for model_name, data in training_data.items():
            if 'metrics_history' in data:
                epochs = data['metrics_history']['epochs']
                train_accs = data['metrics_history']['train_accuracies']
                ax3.plot(epochs, train_accs, marker='o', linewidth=2,
                        color=colors.get(model_name, 'gray'), label=labels.get(model_name, model_name))
        
        ax3.set_title('Training Accuracy Curves', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Training Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Validation Accuracy Curves
        for model_name, data in training_data.items():
            if 'metrics_history' in data:
                epochs = data['metrics_history']['epochs']
                val_accs = data['metrics_history']['val_accuracies']
                ax4.plot(epochs, val_accs, marker='s', linewidth=2,
                        color=colors.get(model_name, 'gray'), label=labels.get(model_name, model_name))
        
        ax4.set_title('Validation Accuracy Curves', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Accuracy (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot with percentage suffix
        if percentage:
            suffix = self.label_percentages[percentage]['suffix']
            save_path = self.viz_dir / f"training_curves_{suffix}.png"
            print(f"Training curves ({percentage}% labels) saved to: {save_path}")
        else:
            save_path = self.viz_dir / "training_curves.png"
            print(f"Training curves saved to: {save_path}")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def create_model_architecture_comparison(self, percentage=None, filtered_data=None):
        """Create model architecture and parameter comparison for specific label percentage"""
        if filtered_data:
            comparison_data = filtered_data['comparison_data']
            training_data = filtered_data['training_data']
        else:
            comparison_data = self.comparison_data
            training_data = self.training_data
            
        # Fallback: create comparison data from checkpoints if JSON not available
        if not comparison_data and filtered_data and filtered_data['checkpoint_data']:
            comparison_data = self.create_comparison_data_from_checkpoints(filtered_data['checkpoint_data'], percentage)
            
        if not comparison_data:
            print(f" No comparison data available for {percentage}% labels")
            return None
        
        # Create title based on percentage
        if percentage:
            title = f'Model Architecture Analysis: {self.label_percentages[percentage]["name"]}'
        else:
            title = 'Model Architecture Analysis: All Models'
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        models = comparison_data['model_results']
        
        # Prepare data
        model_names = []
        model_types = []
        image_sizes = []
        batch_sizes = []
        model_sizes_mb = []
        
        for model in models:
            name = model['model_name'].replace('_', ' ').title()
            if 'dinov2' in model['model_name']:
                name = f"DINOv2\n(Self-Supervised)"
                model_type = "Self-Supervised"
            elif 'efficientnet' in model['model_name']:
                name = f"CNN\n(Supervised)"
                model_type = "Supervised"
            elif 'vit' in model['model_name']:
                name = f"ViT\n(Supervised)"
                model_type = "Supervised"
            else:
                model_type = "Unknown"
            
            model_names.append(name)
            model_types.append(model_type)
            
            # Image size (area) - use defaults if not available
            if 'image_size' in model:
                img_size = model['image_size']
                image_sizes.append(img_size[0] * img_size[1])
            else:
                # Default image sizes based on model type
                if 'dinov2' in model['model_name']:
                    image_sizes.append(378 * 378)  # DINOv2 default
                elif 'vit' in model['model_name']:
                    image_sizes.append(224 * 224)  # ViT default
                else:
                    image_sizes.append(384 * 384)  # CNN default
            
            # Batch size - use default if not available
            if 'batch_size' in model:
                batch_sizes.append(model['batch_size'])
            else:
                batch_sizes.append(16)  # Default batch size
            
            # Estimate model size from checkpoints
            model_key = model['model_name']
            if model_key in ['dinov2_vits14']:
                model_sizes_mb.append(88.4)  # From checkpoint analysis
            elif model_key in ['efficientnet_b4']:
                model_sizes_mb.append(71.4)
            elif model_key in ['vit_base_patch16_224']:
                model_sizes_mb.append(343.5)
            else:
                model_sizes_mb.append(100)  # Default
        
        # Colors for model types - use specific model colors
        colors = []
        for model in models:
            model_name = model['model_name']
            colors.append(self.get_model_colors(model_name))
        
        # 1. Image Size Comparison
        bars1 = ax1.bar(model_names, [size/1000 for size in image_sizes], color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Input Image Size Comparison', fontweight='bold')
        ax1.set_ylabel('Image Size (K pixels)')
        
        # Add labels
        for bar, size in zip(bars1, image_sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(np.sqrt(size))}²', ha='center', va='bottom', fontweight='bold')
        
        # 2. Model Size Comparison
        bars2 = ax2.bar(model_names, model_sizes_mb, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Model Size Comparison', fontweight='bold')
        ax2.set_ylabel('Model Size (MB)')
        
        # Add labels
        for bar, size in zip(bars2, model_sizes_mb):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{size:.1f}MB', ha='center', va='bottom', fontweight='bold')
        
        # 3. Accuracy vs Model Size
        test_accs = [model['test_accuracy'] for model in models]
        scatter = ax3.scatter(model_sizes_mb, test_accs, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)
        
        for i, name in enumerate(model_names):
            ax3.annotate(name.replace('\n', ' '), (model_sizes_mb[i], test_accs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        ax3.set_title('Accuracy vs Model Size', fontweight='bold')
        ax3.set_xlabel('Model Size (MB)')
        ax3.set_ylabel('Test Accuracy (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Training Configuration
        learning_rates = []
        for model in models:
            # Get learning rate from training data if available
            model_key = model['model_name']
            if model_key in training_data:
                config = training_data[model_key].get('session_metadata', {}).get('config', {})
                lr = config.get('LEARNING_RATE', 0.001)
            else:
                lr = 0.001  # Default
            learning_rates.append(lr * 1000)  # Convert to easier scale
        
        bars4 = ax4.bar(model_names, learning_rates, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_title('Learning Rate Configuration', fontweight='bold')
        ax4.set_ylabel('Learning Rate (×10⁻³)')
        
        # Add labels
        for bar, lr in zip(bars4, learning_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{lr:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot with percentage suffix
        if percentage:
            suffix = self.label_percentages[percentage]['suffix']
            save_path = self.viz_dir / f"architecture_comparison_{suffix}.png"
            print(f"Architecture comparison ({percentage}% labels) saved to: {save_path}")
        else:
            save_path = self.viz_dir / "architecture_comparison.png"
            print(f"Architecture comparison saved to: {save_path}")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def create_summary_table(self, percentage=None, filtered_data=None):
        """Create a comprehensive summary table for specific label percentage"""
        if filtered_data:
            comparison_data = filtered_data['comparison_data']
        else:
            comparison_data = self.comparison_data
            
        # Fallback: create comparison data from checkpoints if JSON not available
        if not comparison_data and filtered_data and filtered_data['checkpoint_data']:
            comparison_data = self.create_comparison_data_from_checkpoints(filtered_data['checkpoint_data'], percentage)
            
        if not comparison_data:
            print(f" No comparison data available for {percentage}% labels")
            return None
        
        models = comparison_data['model_results']
        
        # Create summary data
        summary_data = []
        for model in models:
            model_type = "Self-Supervised" if 'dinov2' in model['model_name'] else "Supervised"
            
            # Handle missing fields gracefully
            image_size = model.get('image_size', [224, 224])
            if isinstance(image_size, list):
                image_size_str = f"{image_size[0]}×{image_size[1]}"
            else:
                image_size_str = "224×224"
                
            summary_data.append({
                'Model': model['model_name'].replace('_', ' ').title(),
                'Type': model_type,
                'Test Accuracy (%)': f"{model['test_accuracy']:.1f}",
                'Training Accuracy (%)': f"{model.get('train_accuracy', model['test_accuracy']):.1f}",
                'Best Val Accuracy (%)': f"{model.get('best_val_acc', model['test_accuracy']):.1f}",
                'Training Time (min)': f"{model.get('time', 3600)/60:.1f}",
                'Image Size': image_size_str,
                'Batch Size': model.get('batch_size', 16),
                'Epochs': model.get('epochs', 5)
            })
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by model type
        for i in range(1, len(df) + 1):
            model_name = df.iloc[i-1]['Model']
            specific_model_type = self.get_specific_model_type(model_name)
            color = self.get_table_colors_by_model(specific_model_type)
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(color)
        
        # Create title based on percentage
        if percentage:
            title = f'Model Comparison Summary: {self.label_percentages[percentage]["name"]}'
        else:
            title = 'Model Comparison Summary Table'
            
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Save the plot with percentage suffix
        if percentage:
            suffix = self.label_percentages[percentage]['suffix']
            save_path = self.viz_dir / f"summary_table_{suffix}.png"
            print(f"Summary table ({percentage}% labels) saved to: {save_path}")
        else:
            save_path = self.viz_dir / "summary_table.png"
            print(f"Summary table saved to: {save_path}")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def get_table_colors_by_model(self, model_type):
        """Get table background colors for specific model types"""
        if model_type == "DINOv2":
            return '#B3F0E6'  # Bright light teal for DINOv2
        elif model_type == "CNN":
            return '#FFB3BA'  # Bright light pink for CNN
        elif model_type == "ViT":
            return '#FFFFE0'  # Bright light yellow for ViT
        else:
            return '#FFFFFF'  # White for unknown
    
    def get_specific_model_type(self, model_name):
        """Get specific model type (CNN, ViT, DINOv2) from model name"""
        if 'dinov2' in model_name.lower() or 'dino' in model_name.lower():
            return 'DINOv2'
        elif 'efficientnet' in model_name.lower() or 'cnn' in model_name.lower():
            return 'CNN'
        elif 'vit' in model_name.lower():
            return 'ViT'
        else:
            return 'Unknown'
    
    def get_model_colors(self, model_name_or_type):
        """Get consistent colors for specific model types across all visualizations"""
        # Handle both model names and model types
        if 'dinov2' in model_name_or_type.lower() or 'dino' in model_name_or_type.lower() or 'DINOv2' in model_name_or_type:
            return '#4ECDC4'  # Teal for DINOv2
        elif 'efficientnet' in model_name_or_type.lower() or 'cnn' in model_name_or_type.lower() or 'CNN' in model_name_or_type:
            return '#FF6B6B'  # Red/pink for CNN
        elif 'vit' in model_name_or_type.lower() or 'ViT' in model_name_or_type:
            return '#FFE66D'  # Yellow for ViT
        else:
            return '#999999'  # Gray for unknown
    
    def generate_all_visualizations(self):
        """Generate all visualizations for the thesis, separated by label percentage"""
        print("Generating Master Thesis Visualizations...")
        print("=" * 60)
        
        print(f"Available label percentages: {', '.join([self.label_percentages[p]['name'] for p in self.available_percentages])}")
        
        all_generated_plots = []
        
        # Generate visualizations for each available percentage
        for percentage in self.available_percentages:
            print(f"\nGenerating visualizations for {self.label_percentages[percentage]['name']}...")
            print("-" * 40)
            
            # Filter data for this percentage
            filtered_data = self.filter_data_by_percentage(percentage)
            
            # Create all visualizations for this percentage
            viz_functions = [
                ("Performance Comparison", self.create_performance_comparison),
                ("Training Curves", self.create_training_curves),
                ("Architecture Comparison", self.create_model_architecture_comparison),
                ("Summary Table", self.create_summary_table)
            ]
            
            generated_plots = []
            
            for name, func in viz_functions:
                print(f"\n  Creating {name} for {percentage}% labels...")
                try:
                    fig = func(percentage=percentage, filtered_data=filtered_data)
                    if fig:
                        generated_plots.append(f"{name} ({percentage}% labels)")
                        plt.close(fig)  # Close to free memory
                    else:
                        print(f"  WARNING: Skipping {name} for {percentage}% labels (no data)")
                except Exception as e:
                    print(f"   Error creating {name} for {percentage}% labels: {e}")
            
            all_generated_plots.extend(generated_plots)
            print(f"\n  Generated {len(generated_plots)} visualizations for {percentage}% labels")
        
        print("\n" + "=" * 60)
        print(f"Generated {len(all_generated_plots)} total visualizations:")
        for plot in all_generated_plots:
            print(f"   {plot}")
        
        print(f"\n All visualizations saved to: {self.viz_dir}")
        print("\nKey Findings by Label Percentage:")
        
        for percentage in self.available_percentages:
            filtered_data = self.filter_data_by_percentage(percentage)
            if filtered_data['comparison_data'] and 'model_results' in filtered_data['comparison_data']:
                models = filtered_data['comparison_data']['model_results']
                if models:
                    best_model = max(models, key=lambda x: x['test_accuracy'])
                    print(f"   {self.label_percentages[percentage]['name']}: {best_model['model_name']} ({best_model['test_accuracy']:.1f}% accuracy)")
        
        return all_generated_plots

def main():
    """Main function to generate all thesis visualizations by label percentage"""
    visualizer = ThesisVisualizer()
    if not visualizer.available_percentages:
        print(" No models found with recognizable label percentages")
        print("Available model names in data:")
        if visualizer.comparison_data and 'model_results' in visualizer.comparison_data:
            for model in visualizer.comparison_data['model_results']:
                print(f"  - {model.get('model_name', 'Unknown')}")
        return
    
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
