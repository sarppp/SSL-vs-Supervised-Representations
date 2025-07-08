import torch
import torch.nn as nn
import time
import os
import sys
import numpy as np
import psutil
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import config_paths
from src.models import model_setup
from src.config import config as config_cnn
from src.config import config_dinov2
from src.data.dataloader_setup import create_dataloaders
from src.data import data_splitter

# Set quantization backend
torch.backends.quantized.engine = 'fbgemm'

class ResearchQuantizer:
    """Comprehensive quantization analysis for research purposes"""
    
    def __init__(self, save_dir="quantized_models"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        print(f"[INIT] Research quantizer initialized. Results will be saved to: {self.save_dir.absolute()}")
        
    def load_original_model(self, model_path, config_module, model_name):
        """Load and prepare original model for quantization"""
        print(f"[INFO] Loading {model_name} from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        num_classes = len(checkpoint['class_names'])
        
        # Create model
        model = model_setup.create_model(num_classes, model_name, config_module)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"[INFO] Model loaded successfully. Classes: {num_classes}")
        return model, checkpoint['class_names']
    
    def quantize_cnn(self, model, method='dynamic'):
        """Comprehensive CNN quantization with multiple methods"""
        print(f"[QUANT] Applying {method} quantization to CNN...")
        model.eval()
        model.cpu()
        
        if method == 'dynamic':
            # Dynamic quantization on Linear and Conv2d layers for best performance
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
        elif method == 'static':
            # Static quantization (would need calibration data in production)
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            quantized_model = torch.quantization.convert(model, inplace=False)
        elif method == 'qat':  # Quantization Aware Training (future enhancement)
            # This would require retraining - placeholder for research
            raise NotImplementedError("QAT requires model retraining - use post-training methods")
        else:
            raise ValueError(f"Unknown quantization method: {method}")
            
        print(f"[QUANT] {method.upper()} quantization completed for CNN")
        return quantized_model
    
    def quantize_dino(self, model, method='dynamic'):
        """Comprehensive DINOv2 quantization optimized for ViT architecture"""
        print(f"[QUANT] Applying {method} quantization to DINOv2...")
        model.eval()
        model.cpu()
        
        if method == 'dynamic':
            # For ViT, Linear layers are the computational bottleneck
            # Focus on Linear layers for maximum efficiency gain
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
        elif method == 'static':
            # Static quantization for ViT (experimental)
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            quantized_model = torch.quantization.convert(model, inplace=False)
        else:
            raise ValueError(f"Unknown quantization method: {method}")
            
        print(f"[QUANT] {method.upper()} quantization completed for DINOv2")
        return quantized_model
    
    def comprehensive_evaluation(self, model, dataloader, device='cpu', name="Model", full_dataset=True):
        """Comprehensive performance evaluation for research analysis"""
        print(f"[EVAL] Comprehensive evaluation of {name}...")
        
        # Handle quantized models - they must run on CPU
        if hasattr(model, 'qconfig') and model.qconfig is not None:
            device = 'cpu'
            print(f"[EVAL] Quantized model detected - using CPU")
        else:
            model.to(device)
            print(f"[EVAL] Original model - using {device}")
        
        model.eval()
        
        # Detailed metrics tracking
        correct, total = 0, 0
        inference_times = []
        batch_times = []
        class_correct = {}
        class_total = {}
        
        # Memory measurement setup
        if device == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1e6
        else:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1e6
        
        peak_memory = initial_memory
        
        # Comprehensive evaluation
        max_batches = len(dataloader) if full_dataset else min(50, len(dataloader))
        print(f"[EVAL] Evaluating on {max_batches} batches (full_dataset={full_dataset})")
        
        with torch.inference_mode():
            for i, (images, labels) in enumerate(tqdm(dataloader, total=max_batches, desc=f"Evaluating {name}")):
                batch_start = time.time()
                
                # Handle quantized models
                if hasattr(model, 'qconfig') and model.qconfig is not None:
                    images, labels = images.cpu(), labels.cpu()
                else:
                    images, labels = images.to(device), labels.to(device)
                
                # Measure inference time per batch
                inference_start = time.time()
                outputs = model(images)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time / images.size(0))  # Per image
                
                # Batch processing time
                batch_times.append(time.time() - batch_start)
                
                # Calculate accuracy
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Per-class accuracy tracking
                for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                    if label not in class_correct:
                        class_correct[label] = 0
                        class_total[label] = 0
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
                
                # Memory tracking
                if device == 'cuda':
                    current_memory = torch.cuda.memory_allocated() / 1e6
                else:
                    current_memory = process.memory_info().rss / 1e6
                peak_memory = max(peak_memory, current_memory)
                
                # Break if not full dataset evaluation
                if i >= max_batches - 1:
                    break
        
        # Calculate comprehensive metrics
        accuracy = correct / total if total > 0 else 0.0
        avg_inference_time = np.mean(inference_times) * 1000  # ms per image
        std_inference_time = np.std(inference_times) * 1000   # ms per image
        avg_batch_time = np.mean(batch_times) * 1000         # ms per batch
        memory_usage = peak_memory - initial_memory
        
        # Per-class accuracy
        class_accuracies = {}
        for class_id in class_total:
            class_accuracies[class_id] = class_correct[class_id] / class_total[class_id]
        
        # Calculate additional research metrics
        throughput = total / sum(batch_times) if sum(batch_times) > 0 else 0  # images/second
        
        metrics = {
            'accuracy': accuracy,
            'total_samples': total,
            'correct_predictions': correct,
            'inference_time_ms_mean': avg_inference_time,
            'inference_time_ms_std': std_inference_time,
            'batch_time_ms_mean': avg_batch_time,
            'memory_usage_mb': memory_usage,
            'peak_memory_mb': peak_memory,
            'throughput_imgs_per_sec': throughput,
            'class_accuracies': class_accuracies,
            'num_classes_evaluated': len(class_accuracies),
            'evaluation_batches': min(i + 1, max_batches)
        }
        
        print(f"[EVAL] {name} - Accuracy: {accuracy:.1%}, Speed: {avg_inference_time:.2f}±{std_inference_time:.2f}ms, Memory: {memory_usage:.1f}MB")
        return metrics
    
    def get_model_size_mb(self, model):
        """Get model size in MB with detailed breakdown"""
        temp_path = "temp_model.pth"
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / 1e6
        os.remove(temp_path)
        
        # Count parameters for additional analysis
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'size_mb': size_mb,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameters_millions': total_params / 1e6
        }
    
    def calculate_research_metrics(self, original_metrics, quantized_metrics, original_size_info, quantized_size_info):
        """Calculate comprehensive research metrics for comparison"""
        
        # Performance preservation
        accuracy_retention = quantized_metrics['accuracy'] / original_metrics['accuracy'] if original_metrics['accuracy'] > 0 else 0
        
        # Efficiency gains
        size_reduction = original_size_info['size_mb'] / quantized_size_info['size_mb'] if quantized_size_info['size_mb'] > 0 else 1
        speed_improvement = original_metrics['inference_time_ms_mean'] / quantized_metrics['inference_time_ms_mean'] if quantized_metrics['inference_time_ms_mean'] > 0 else 1
        memory_reduction = original_metrics['memory_usage_mb'] / quantized_metrics['memory_usage_mb'] if quantized_metrics['memory_usage_mb'] > 0 else 1
        
        # Throughput improvement
        throughput_improvement = quantized_metrics['throughput_imgs_per_sec'] / original_metrics['throughput_imgs_per_sec'] if original_metrics['throughput_imgs_per_sec'] > 0 else 1
        
        # Parameter reduction
        param_reduction = original_size_info['total_parameters'] / quantized_size_info['total_parameters'] if quantized_size_info['total_parameters'] > 0 else 1
        
        # Research-grade drone deployment score
        drone_score = self.calculate_research_drone_score(quantized_metrics, quantized_size_info)
        
        return {
            'accuracy_retention_percent': accuracy_retention * 100,
            'size_reduction_factor': size_reduction,
            'speed_improvement_factor': speed_improvement,
            'memory_reduction_factor': memory_reduction,
            'throughput_improvement_factor': throughput_improvement,
            'parameter_reduction_factor': param_reduction,
            'drone_deployment_score': drone_score,
            'efficiency_score': (size_reduction + speed_improvement + memory_reduction) / 3 * accuracy_retention
        }
    
    def calculate_research_drone_score(self, metrics, size_info):
        """Research-grade drone deployment suitability score"""
        accuracy = metrics['accuracy']
        speed_ms = metrics['inference_time_ms_mean']
        memory_mb = metrics['memory_usage_mb']
        model_size_mb = size_info['size_mb']
        
        # Research-based scoring with more nuanced thresholds
        accuracy_score = accuracy * 100
        
        # Speed scoring: ideal <5ms, acceptable <10ms, poor >15ms
        if speed_ms <= 5:
            speed_score = 100
        elif speed_ms <= 10:
            speed_score = 100 - (speed_ms - 5) * 10
        elif speed_ms <= 15:
            speed_score = 50 - (speed_ms - 10) * 5
        else:
            speed_score = max(0, 25 - (speed_ms - 15) * 2)
        
        # Memory scoring: ideal <50MB, acceptable <100MB, poor >150MB
        if memory_mb <= 50:
            memory_score = 100
        elif memory_mb <= 100:
            memory_score = 100 - (memory_mb - 50) * 1
        elif memory_mb <= 150:
            memory_score = 50 - (memory_mb - 100) * 0.5
        else:
            memory_score = max(0, 25 - (memory_mb - 150) * 0.2)
        
        # Size scoring: ideal <25MB, acceptable <50MB, poor >100MB
        if model_size_mb <= 25:
            size_score = 100
        elif model_size_mb <= 50:
            size_score = 100 - (model_size_mb - 25) * 2
        elif model_size_mb <= 100:
            size_score = 50 - (model_size_mb - 50) * 0.5
        else:
            size_score = max(0, 25 - (model_size_mb - 100) * 0.1)
        
        # Weighted research score
        research_drone_score = (accuracy_score * 0.4 + 
                               speed_score * 0.3 + 
                               memory_score * 0.2 + 
                               size_score * 0.1)
        
        return min(100, max(0, research_drone_score))
    
    def save_research_results(self, model, model_name, quant_type, metrics, original_metrics, research_metrics, class_names, size_info):
        """Save comprehensive research results with metadata"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"quant_{model_name}_{quant_type.lower()}_{timestamp}.pth"
        filepath = self.save_dir / filename
        
        # Comprehensive save data for research
        save_data = {
            # Model data
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'quantization_type': quant_type,
            'class_names': class_names,
            
            # Performance metrics
            'quantized_metrics': metrics,
            'original_metrics': original_metrics,
            'research_metrics': research_metrics,
            
            # Model size information
            'size_info': size_info,
            
            # Research metadata
            'evaluation_timestamp': timestamp,
            'pytorch_version': torch.__version__,
            'quantization_backend': torch.backends.quantized.engine,
            'device_info': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'cpu_count': os.cpu_count()
            },
            
            # Research notes
            'research_notes': {
                'purpose': 'Agricultural drone deployment research',
                'quantization_focus': 'Dynamic quantization for real-time inference',
                'evaluation_methodology': 'Comprehensive accuracy and efficiency analysis'
            }
        }
        
        torch.save(save_data, filepath)
        print(f"[SAVE] Research results saved: {filename}")
        print(f"[SAVE] Size: {size_info['size_mb']:.1f}MB, Compression: {research_metrics['size_reduction_factor']:.1f}x")
        return filepath

def main():
    print("🔬 RESEARCH-GRADE QUANTIZATION ANALYSIS")
    print("=" * 70)
    print("Purpose: Comprehensive quantization study for agricultural drone deployment")
    print("Focus: Accuracy preservation vs. efficiency gains")
    print("=" * 70)
    
    # Initialize research quantizer
    quantizer = ResearchQuantizer()
    
    # Device configuration for research
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[CONFIG] Device: {device}")
    print(f"[CONFIG] PyTorch version: {torch.__version__}")
    print(f"[CONFIG] Quantization backend: {torch.backends.quantized.engine}")
    
    # Research-appropriate batch sizes (not optimized for speed)
    if device == 'cuda':
        config_cnn.BATCH_SIZE = 16  # Conservative for thorough evaluation
        config_dinov2.BATCH_SIZE = 16
        print(f"[CONFIG] Using GPU with batch size 16 for thorough evaluation")
    else:
        config_cnn.BATCH_SIZE = 8
        config_dinov2.BATCH_SIZE = 8
        print(f"[CONFIG] Using CPU with batch size 8 for thorough evaluation")
    
    # Ensure adequate workers for research
    cpu_count = os.cpu_count() or 4  # Default to 4 if cpu_count() returns None
    config_cnn.NUM_WORKERS = min(4, cpu_count)
    config_dinov2.NUM_WORKERS = min(4, cpu_count)
    
    # Load complete dataset for research accuracy
    print("\n[DATA] Loading complete dataset for research analysis...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = data_splitter.split_clean_dataset(
        pickle_path=config_paths.CLEAN_DATASET_PICKLE,
        base_data_dir=config_paths.BASE_DATA_DIR,
        few_shot_mode=None
    )
    
    all_labels = train_labels + val_labels + test_labels
    total_samples = len(train_paths) + len(val_paths) + len(test_paths)
    
    # For research: use substantial portion or full dataset
    use_full_dataset = total_samples < 10000  # Use full if reasonable size
    if use_full_dataset:
        print(f"[DATA] Using complete dataset: {total_samples:,} samples")
        sample_paths = train_paths + val_paths + test_paths
        sample_labels = train_labels + val_labels + test_labels
    else:
        # Use 20% for comprehensive research if dataset is very large
        sample_size = int(total_samples * 0.2)
        print(f"[DATA] Using research subset: {sample_size:,} samples ({sample_size/total_samples*100:.1f}% of {total_samples:,})")
        
        # Stratified sampling to maintain class distribution
        np.random.seed(42)
        all_paths = train_paths + val_paths + test_paths
        indices = np.random.choice(len(all_paths), size=sample_size, replace=False)
        sample_paths = [all_paths[i] for i in indices]
        sample_labels = [all_labels[i] for i in indices]
    
    # Split sampled data for proper evaluation
    from sklearn.model_selection import train_test_split
    train_sample_paths, test_sample_paths, train_sample_labels, test_sample_labels = train_test_split(
        sample_paths, sample_labels, test_size=0.3, random_state=42, stratify=sample_labels
    )
    val_sample_paths, test_sample_paths, val_sample_labels, test_sample_labels = train_test_split(
        test_sample_paths, test_sample_labels, test_size=0.5, random_state=42, stratify=test_sample_labels
    )
    
    print(f"[DATA] Research split - Train: {len(train_sample_paths):,}, Val: {len(val_sample_paths):,}, Test: {len(test_sample_paths):,}")
    print(f"[DATA] Classes in dataset: {len(set(sample_labels))}")
    
    # Create dataloaders for research
    print("[DATA] Creating research dataloaders...")
    cnn_train_loader, cnn_val_loader, cnn_test_loader, *_ = create_dataloaders(
        train_sample_paths, train_sample_labels, val_sample_paths, val_sample_labels, 
        test_sample_paths, test_sample_labels, config_module=config_cnn
    )
    
    dino_train_loader, dino_val_loader, dino_test_loader, *_ = create_dataloaders(
        train_sample_paths, train_sample_labels, val_sample_paths, val_sample_labels,
        test_sample_paths, test_sample_labels, config_module=config_dinov2
    )
    
    # Model paths (research models)
    cnn_path = '/home/models/best_cnn_b4_acc76.06_20250703_124117.pth'
    dino_path = '/home/models/best_dino_vits14_acc75.00_20250703_124256.pth'
    
    # Load original models for research
    print("\n[MODELS] Loading research models...")
    cnn_original, class_names = quantizer.load_original_model(cnn_path, config_cnn, 'efficientnet_b4')
    dino_original, _ = quantizer.load_original_model(dino_path, config_dinov2, 'dinov2_vits14')
    
    # Comprehensive evaluation of original models
    print("\n[RESEARCH] Comprehensive evaluation of original models...")
    cnn_original_metrics = quantizer.comprehensive_evaluation(
        cnn_original, cnn_test_loader, device, "CNN Original", full_dataset=True
    )
    dino_original_metrics = quantizer.comprehensive_evaluation(
        dino_original, dino_test_loader, device, "DINOv2 Original", full_dataset=True
    )
    
    # Get detailed size information
    cnn_original_size = quantizer.get_model_size_mb(cnn_original)
    dino_original_size = quantizer.get_model_size_mb(dino_original)
    
    print(f"[RESEARCH] Original CNN - {cnn_original_size['parameters_millions']:.1f}M parameters, {cnn_original_size['size_mb']:.1f}MB")
    print(f"[RESEARCH] Original DINOv2 - {dino_original_size['parameters_millions']:.1f}M parameters, {dino_original_size['size_mb']:.1f}MB")
    
    # Research quantization process
    print("\n[RESEARCH] Applying quantization methods...")
    
    # CNN Dynamic Quantization
    print("\n[QUANT] CNN Dynamic Quantization Research...")
    start_time = time.time()
    quant_cnn_dynamic = quantizer.quantize_cnn(cnn_original, method='dynamic')
    cnn_quant_time = time.time() - start_time
    print(f"[TIMING] CNN quantization completed in {cnn_quant_time:.2f}s")
    
    # DINOv2 Dynamic Quantization  
    print("\n[QUANT] DINOv2 Dynamic Quantization Research...")
    start_time = time.time()
    quant_dino_dynamic = quantizer.quantize_dino(dino_original, method='dynamic')
    dino_quant_time = time.time() - start_time
    print(f"[TIMING] DINOv2 quantization completed in {dino_quant_time:.2f}s")
    
    # Comprehensive evaluation of quantized models
    print("\n[RESEARCH] Comprehensive evaluation of quantized models...")
    cnn_quant_metrics = quantizer.comprehensive_evaluation(
        quant_cnn_dynamic, cnn_test_loader, 'cpu', "CNN Quantized", full_dataset=True
    )
    dino_quant_metrics = quantizer.comprehensive_evaluation(
        quant_dino_dynamic, dino_test_loader, 'cpu', "DINOv2 Quantized", full_dataset=True
    )
    
    # Get quantized model size information
    cnn_quant_size = quantizer.get_model_size_mb(quant_cnn_dynamic)
    dino_quant_size = quantizer.get_model_size_mb(quant_dino_dynamic)
    
    # Calculate comprehensive research metrics
    print("\n[RESEARCH] Calculating comprehensive research metrics...")
    cnn_research_metrics = quantizer.calculate_research_metrics(
        cnn_original_metrics, cnn_quant_metrics, cnn_original_size, cnn_quant_size
    )
    dino_research_metrics = quantizer.calculate_research_metrics(
        dino_original_metrics, dino_quant_metrics, dino_original_size, dino_quant_size
    )
    
    # Save comprehensive research results
    print("\n[SAVE] Saving comprehensive research results...")
    cnn_save_path = quantizer.save_research_results(
        quant_cnn_dynamic, 'cnn', 'Dynamic', cnn_quant_metrics, cnn_original_metrics, 
        cnn_research_metrics, class_names, cnn_quant_size
    )
    dino_save_path = quantizer.save_research_results(
        quant_dino_dynamic, 'dino', 'Dynamic', dino_quant_metrics, dino_original_metrics,
        dino_research_metrics, class_names, dino_quant_size
    )
    
    # Comprehensive Research Results
    print("\n" + "=" * 90)
    print("📊 COMPREHENSIVE RESEARCH RESULTS")
    print("=" * 90)
    
    # Detailed comparison table
    print(f"\n🔬 DETAILED PERFORMANCE ANALYSIS")
    print(f"{'Model':<25} | {'Accuracy':>8} | {'Speed (ms)':>11} | {'±StdDev':>8} | {'Memory (MB)':>11} | {'Throughput':>10} | {'Drone Score':>11}")
    print("-" * 100)
    print(f"{'CNN Original':<25} | {cnn_original_metrics['accuracy']:7.1%} | {cnn_original_metrics['inference_time_ms_mean']:10.2f} | {cnn_original_metrics['inference_time_ms_std']:7.2f} | {cnn_original_metrics['memory_usage_mb']:10.1f} | {cnn_original_metrics['throughput_imgs_per_sec']:9.1f} | {quantizer.calculate_research_drone_score(cnn_original_metrics, cnn_original_size):10.1f}")
    print(f"{'CNN Quantized':<25} | {cnn_quant_metrics['accuracy']:7.1%} | {cnn_quant_metrics['inference_time_ms_mean']:10.2f} | {cnn_quant_metrics['inference_time_ms_std']:7.2f} | {cnn_quant_metrics['memory_usage_mb']:10.1f} | {cnn_quant_metrics['throughput_imgs_per_sec']:9.1f} | {cnn_research_metrics['drone_deployment_score']:10.1f}")
    print(f"{'DINOv2 Original':<25} | {dino_original_metrics['accuracy']:7.1%} | {dino_original_metrics['inference_time_ms_mean']:10.2f} | {dino_original_metrics['inference_time_ms_std']:7.2f} | {dino_original_metrics['memory_usage_mb']:10.1f} | {dino_original_metrics['throughput_imgs_per_sec']:9.1f} | {quantizer.calculate_research_drone_score(dino_original_metrics, dino_original_size):10.1f}")
    print(f"{'DINOv2 Quantized':<25} | {dino_quant_metrics['accuracy']:7.1%} | {dino_quant_metrics['inference_time_ms_mean']:10.2f} | {dino_quant_metrics['inference_time_ms_std']:7.2f} | {dino_quant_metrics['memory_usage_mb']:10.1f} | {dino_quant_metrics['throughput_imgs_per_sec']:9.1f} | {dino_research_metrics['drone_deployment_score']:10.1f}")
    
    # Research insights
    print(f"\n📈 QUANTIZATION RESEARCH INSIGHTS")
    print(f"CNN Results:")
    print(f"  • Accuracy Retention: {cnn_research_metrics['accuracy_retention_percent']:.1f}%")
    print(f"  • Size Reduction: {cnn_research_metrics['size_reduction_factor']:.1f}x ({cnn_original_size['size_mb']:.1f}MB → {cnn_quant_size['size_mb']:.1f}MB)")
    print(f"  • Speed Improvement: {cnn_research_metrics['speed_improvement_factor']:.1f}x")
    print(f"  • Memory Efficiency: {cnn_research_metrics['memory_reduction_factor']:.1f}x improvement")
    print(f"  • Parameter Reduction: {cnn_research_metrics['parameter_reduction_factor']:.1f}x")
    print(f"  • Overall Efficiency Score: {cnn_research_metrics['efficiency_score']:.1f}")
    
    print(f"\nDINOv2 Results:")
    print(f"  • Accuracy Retention: {dino_research_metrics['accuracy_retention_percent']:.1f}%")
    print(f"  • Size Reduction: {dino_research_metrics['size_reduction_factor']:.1f}x ({dino_original_size['size_mb']:.1f}MB → {dino_quant_size['size_mb']:.1f}MB)")
    print(f"  • Speed Improvement: {dino_research_metrics['speed_improvement_factor']:.1f}x")
    print(f"  • Memory Efficiency: {dino_research_metrics['memory_reduction_factor']:.1f}x improvement")
    print(f"  • Parameter Reduction: {dino_research_metrics['parameter_reduction_factor']:.1f}x")
    print(f"  • Overall Efficiency Score: {dino_research_metrics['efficiency_score']:.1f}")
    
    # Research recommendations
    print(f"\n🎯 RESEARCH CONCLUSIONS & RECOMMENDATIONS")
    
    # Determine best model based on research metrics
    models_comparison = [
        ('CNN Original', quantizer.calculate_research_drone_score(cnn_original_metrics, cnn_original_size), cnn_original_metrics, cnn_original_size),
        ('CNN Quantized', cnn_research_metrics['drone_deployment_score'], cnn_quant_metrics, cnn_quant_size),
        ('DINOv2 Original', quantizer.calculate_research_drone_score(dino_original_metrics, dino_original_size), dino_original_metrics, dino_original_size),
        ('DINOv2 Quantized', dino_research_metrics['drone_deployment_score'], dino_quant_metrics, dino_quant_size)
    ]
    
    best_model = max(models_comparison, key=lambda x: x[1])
    
    print(f"\n🏆 OPTIMAL MODEL FOR AGRICULTURAL DRONE DEPLOYMENT: {best_model[0]}")
    print(f"   Research Drone Score: {best_model[1]:.1f}/100")
    print(f"   Accuracy: {best_model[2]['accuracy']:.1%}")
    print(f"   Inference Speed: {best_model[2]['inference_time_ms_mean']:.2f}±{best_model[2]['inference_time_ms_std']:.2f}ms")
    print(f"   Memory Usage: {best_model[2]['memory_usage_mb']:.1f}MB")
    print(f"   Model Size: {best_model[3]['size_mb']:.1f}MB")
    print(f"   Throughput: {best_model[2]['throughput_imgs_per_sec']:.1f} images/second")
    
    # Research dataset summary
    print(f"\n📊 RESEARCH DATASET SUMMARY")
    print(f"   Total samples evaluated: {cnn_quant_metrics['total_samples']:,}")
    print(f"   Number of classes: {cnn_quant_metrics['num_classes_evaluated']}")
    print(f"   Evaluation batches: {cnn_quant_metrics['evaluation_batches']}")
    
    # File outputs for research
    print(f"\n💾 RESEARCH OUTPUT FILES")
    print(f"   📁 Directory: {quantizer.save_dir.absolute()}")
    print(f"   📄 CNN Quantized: {cnn_save_path.name}")
    print(f"   📄 DINOv2 Quantized: {dino_save_path.name}")
    print(f"\n📝 RESEARCH NOTES:")
    print(f"   • Models saved with comprehensive metadata for paper writing")
    print(f"   • Per-class accuracy metrics included for detailed analysis")
    print(f"   • Statistical measures (mean, std) provided for reproducibility")
    print(f"   • Device and environment information saved for methodology section")
    
    # Research timing summary
    total_time = cnn_quant_time + dino_quant_time
    print(f"\n⏱️  RESEARCH TIMING SUMMARY")
    print(f"   • CNN quantization: {cnn_quant_time:.2f}s")
    print(f"   • DINOv2 quantization: {dino_quant_time:.2f}s")
    print(f"   • Total quantization time: {total_time:.2f}s")
    print(f"   • Evaluation thoroughness: Comprehensive (full test set)")

if __name__ == "__main__":
    # Add sklearn import for stratified splitting
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Error: scikit-learn is required for research-grade evaluation")
        print("Install with: pip install scikit-learn")
        sys.exit(1)
    
    main() 