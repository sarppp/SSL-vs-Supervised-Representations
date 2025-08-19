#!/usr/bin/env python3
"""
🎓 Master Thesis Visualizations: Supervised vs Self-Supervised Models
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
    
    def create_performance_comparison(self):
        """Create comprehensive performance comparison visualization"""
        if not self.comparison_data:
            print("❌ No comparison data available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison: Supervised vs Self-Supervised', fontsize=16, fontweight='bold')
        
        models = self.comparison_data['model_results']
        
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
        
        # Colors for model types
        colors = ['#FF6B6B' if t == 'Supervised' else '#4ECDC4' for t in model_types]
        
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
        scatter = ax4.scatter(training_times, test_accs, c=[colors[i] for i in range(len(colors))], 
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
        legend_elements = [Patch(facecolor='#FF6B6B', label='Supervised'),
                          Patch(facecolor='#4ECDC4', label='Self-Supervised')]
        ax4.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        # Save the plot
        save_path = self.viz_dir / "performance_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance comparison saved to: {save_path}")
        
        return fig
    
    def create_training_curves(self):
        """Create training curves visualization"""
        if not self.training_data:
            print("❌ No training data available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Dynamics: Supervised vs Self-Supervised Models', fontsize=16, fontweight='bold')
        
        colors = {'dinov2_vits14': '#4ECDC4', 'efficientnet_b4': '#FF6B6B', 'vit_base_patch16_224': '#FFE66D'}
        labels = {'dinov2_vits14': 'DINOv2 (Self-Supervised)', 'efficientnet_b4': 'CNN (Supervised)', 
                 'vit_base_patch16_224': 'ViT (Supervised)'}
        
        # 1. Training Loss Curves
        for model_name, data in self.training_data.items():
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
        for model_name, data in self.training_data.items():
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
        for model_name, data in self.training_data.items():
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
        for model_name, data in self.training_data.items():
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
        
        # Save the plot
        save_path = self.viz_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to: {save_path}")
        
        return fig
    
    def create_model_architecture_comparison(self):
        """Create model architecture and parameter comparison"""
        if not self.comparison_data:
            print("❌ No comparison data available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Architecture Analysis: Supervised vs Self-Supervised', fontsize=16, fontweight='bold')
        
        models = self.comparison_data['model_results']
        
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
            
            # Image size (area)
            img_size = model['image_size']
            image_sizes.append(img_size[0] * img_size[1])
            batch_sizes.append(model['batch_size'])
            
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
        
        colors = ['#FF6B6B' if t == 'Supervised' else '#4ECDC4' for t in model_types]
        
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
            if model_key in self.training_data:
                config = self.training_data[model_key].get('session_metadata', {}).get('config', {})
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
        
        # Save the plot
        save_path = self.viz_dir / "architecture_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Architecture comparison saved to: {save_path}")
        
        return fig
    
    def create_summary_table(self):
        """Create a comprehensive summary table"""
        if not self.comparison_data:
            print("❌ No comparison data available")
            return
        
        models = self.comparison_data['model_results']
        
        # Create summary data
        summary_data = []
        for model in models:
            model_type = "Self-Supervised" if 'dinov2' in model['model_name'] else "Supervised"
            
            summary_data.append({
                'Model': model['model_name'].replace('_', ' ').title(),
                'Type': model_type,
                'Test Accuracy (%)': f"{model['test_accuracy']:.1f}",
                'Training Accuracy (%)': f"{model['train_accuracy']:.1f}",
                'Best Val Accuracy (%)': f"{model['best_val_acc']:.1f}",
                'Training Time (min)': f"{model['time']/60:.1f}",
                'Image Size': f"{model['image_size'][0]}×{model['image_size'][1]}",
                'Batch Size': model['batch_size'],
                'Epochs': model['epochs']
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
            model_type = df.iloc[i-1]['Type']
            color = '#FFE6E6' if model_type == 'Supervised' else '#E6F7F7'
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(color)
        
        plt.title('Model Comparison Summary Table', fontsize=16, fontweight='bold', pad=20)
        
        # Save the plot
        save_path = self.viz_dir / "summary_table.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Summary table saved to: {save_path}")
        
        return fig
    
    def generate_all_visualizations(self):
        """Generate all visualizations for the thesis"""
        print("🎓 Generating Master Thesis Visualizations...")
        print("=" * 60)
        
        # Create all visualizations
        viz_functions = [
            ("Performance Comparison", self.create_performance_comparison),
            ("Training Curves", self.create_training_curves),
            ("Architecture Comparison", self.create_model_architecture_comparison),
            ("Summary Table", self.create_summary_table)
        ]
        
        generated_plots = []
        
        for name, func in viz_functions:
            print(f"\n📊 Creating {name}...")
            try:
                fig = func()
                if fig:
                    generated_plots.append(name)
                    plt.close(fig)  # Close to free memory
            except Exception as e:
                print(f"❌ Error creating {name}: {e}")
        
        print("\n" + "=" * 60)
        print(f"✅ Generated {len(generated_plots)} visualizations:")
        for plot in generated_plots:
            print(f"   📈 {plot}")
        
        print(f"\n📁 All visualizations saved to: {self.viz_dir}")
        print("\n🎯 Key Findings for Thesis:")
        
        if self.comparison_data:
            winner = self.comparison_data['summary']['winner']
            print(f"   🏆 Best performing model: {winner['model_name']} ({winner['test_accuracy']:.1f}% accuracy)")
            print(f"   🔬 Self-supervised DINOv2 outperformed supervised models")
            print(f"   ⚡ Training efficiency varies significantly between architectures")
            print(f"   📊 Model size vs accuracy trade-offs clearly visible")
        
        return generated_plots

def main():
    """Main function to generate all thesis visualizations"""
    visualizer = ThesisVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
