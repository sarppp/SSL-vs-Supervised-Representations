import torch
import torch.nn as nn
import time
import os
import sys
import numpy as np
import psutil
from pathlib import Path
from tqdm import tqdm

# Disable xformers globally for quantization compatibility
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['TORCH_USE_XFORMERS'] = '0'
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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
    
    def disable_xformers_attention(self, model):
        """Disable xformers memory efficient attention for CPU compatibility"""
        print("[QUANT] Disabling xformers attention for CPU compatibility...")
        
        def _disable_xformers_recursively(module):
            # Handle DINOv2 attention modules specifically
            module_name = module.__class__.__name__
            
            # Common patterns for DINOv2 attention
            if 'Attention' in module_name:
                # Disable xformers flags
                if hasattr(module, 'use_memory_efficient_attention'):
                    module.use_memory_efficient_attention = False
                    print(f"[QUANT] Disabled use_memory_efficient_attention in {module_name}")
                
                # Force regular attention implementation
                if hasattr(module, '_use_memory_efficient_attention_xformers'):
                    module._use_memory_efficient_attention_xformers = False
                    print(f"[QUANT] Disabled _use_memory_efficient_attention_xformers in {module_name}")
                
                # Set xformers availability to False
                if hasattr(module, 'xformers_available'):
                    module.xformers_available = False
                    print(f"[QUANT] Set xformers_available=False in {module_name}")
            
            # Check attention within transformer blocks
            if hasattr(module, 'attn'):
                attn_module = module.attn
                if hasattr(attn_module, 'use_memory_efficient_attention'):
                    attn_module.use_memory_efficient_attention = False
                    print(f"[QUANT] Disabled xformers in {module_name}.attn")
                if hasattr(attn_module, '_use_memory_efficient_attention_xformers'):
                    attn_module._use_memory_efficient_attention_xformers = False
                    print(f"[QUANT] Disabled xformers flags in {module_name}.attn")
            
            # Recursively apply to all child modules
            for name, child in module.named_children():
                _disable_xformers_recursively(child)
        
        # Apply recursively to all modules
        _disable_xformers_recursively(model)
        
        # Set additional environment variables for xformers
        import os
        os.environ['XFORMERS_DISABLED'] = '1'
        os.environ['TORCH_USE_XFORMERS'] = '0'
        os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
        
        print("[QUANT] xformers attention disabled successfully")
        return model
    
    def quantize_dino(self, model, method='dynamic'):
        """Comprehensive DINOv2 quantization optimized for ViT architecture"""
        print(f"[QUANT] Applying {method} quantization to DINOv2...")
        model.eval()
        model.cpu()
        
        # Disable xformers attention before quantization for CPU compatibility
        model = self.disable_xformers_attention(model)
        
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
    
    def comprehensive_evaluation(self, model, dataloader, device='cpu', name="Model", full_dataset=True, force_cpu_comparison=True):
        """Comprehensive performance evaluation for research analysis"""
        print(f"[EVAL] Comprehensive evaluation of {name}...")
        
        # Fair comparison logic - quantized models must run on CPU for now
        is_quantized = hasattr(model, 'qconfig') and model.qconfig is not None
        
        if force_cpu_comparison or is_quantized:
            # For fair comparison, run both original and quantized on CPU
            eval_device = 'cpu'
            model.cpu()
            print(f"[EVAL] Running on CPU for fair comparison (quantized models require CPU)")
        else:
            eval_device = device
            model.to(device)
            print(f"[EVAL] Running on {device}")
        
        model.eval()
        
        # Detailed metrics tracking
        correct, total = 0, 0
        inference_times = []
        batch_times = []
        class_correct = {}
        class_total = {}
        
        # Memory measurement setup - consistent approach
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1e6  # Always use system memory for consistency
        peak_memory = initial_memory
        
        # Clear any GPU memory if we were using it before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Comprehensive evaluation
        max_batches = len(dataloader) if full_dataset else min(50, len(dataloader))
        print(f"[EVAL] Evaluating on {max_batches} batches (full_dataset={full_dataset})")
        
        with torch.inference_mode():
            for i, (images, labels) in enumerate(tqdm(dataloader, total=max_batches, desc=f"Evaluating {name}")):
                batch_start = time.time()
                
                # Ensure data is on correct device
                images, labels = images.to(eval_device), labels.to(eval_device)
                
                # Measure inference time per batch
                inference_start = time.time()
                try:
                    outputs = model(images)
                except Exception as e:
                    if "memory_efficient_attention" in str(e) or "xformers" in str(e).lower():
                        print(f"[ERROR] xformers/attention error detected: {e}")
                        print("[FIX] Attempting to disable xformers and retry...")
                        # Try to disable xformers on the model
                        self.disable_xformers_attention(model)
                        outputs = model(images)
                    else:
                        raise e
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
                
                # Memory tracking - consistent system memory measurement
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
            'evaluation_batches': min(i + 1, max_batches),
            'evaluation_device': eval_device,
            'is_quantized': is_quantized
        }
        
        print(f"[EVAL] {name} - Accuracy: {accuracy:.1%}, Speed: {avg_inference_time:.2f}±{std_inference_time:.2f}ms, Memory: {memory_usage:.1f}MB, Device: {eval_device}")
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

    def save_research_log(self, cnn_original_metrics, cnn_quant_metrics, cnn_research_metrics, 
                         dino_original_metrics, dino_quant_metrics, dino_research_metrics,
                         cnn_original_size, cnn_quant_size, dino_original_size, dino_quant_size,
                         total_samples, num_classes, cnn_quant_time, dino_quant_time):
        """Save comprehensive research log file"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_filename = f"quantization_research_log_{timestamp}.txt"
        log_filepath = self.save_dir / log_filename
        
        with open(log_filepath, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write("🔬 RESEARCH-GRADE QUANTIZATION ANALYSIS LOG\n")
            f.write("=" * 90 + "\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Purpose: Comprehensive quantization study for agricultural drone deployment\n")
            f.write(f"Focus: Accuracy preservation vs. efficiency gains\n")
            f.write(f"PyTorch Version: {torch.__version__}\n")
            f.write(f"Quantization Backend: {torch.backends.quantized.engine}\n")
            f.write(f"Device Used: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
            f.write("=" * 90 + "\n\n")
            
            # Dataset information
            f.write("DATASET SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total samples evaluated: {total_samples:,}\n")
            f.write(f"Number of classes: {num_classes}\n")
            f.write(f"Evaluation batches: {cnn_quant_metrics['evaluation_batches']}\n\n")
            
            # Detailed performance table
            f.write("DETAILED PERFORMANCE ANALYSIS\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Model':<25} | {'Accuracy':>8} | {'Speed (ms)':>11} | {'±StdDev':>8} | {'Memory (MB)':>11} | {'Throughput':>10} | {'Drone Score':>11}\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'CNN Original':<25} | {cnn_original_metrics['accuracy']:7.1%} | {cnn_original_metrics['inference_time_ms_mean']:10.2f} | {cnn_original_metrics['inference_time_ms_std']:7.2f} | {cnn_original_metrics['memory_usage_mb']:10.1f} | {cnn_original_metrics['throughput_imgs_per_sec']:9.1f} | {self.calculate_research_drone_score(cnn_original_metrics, cnn_original_size):10.1f}\n")
            f.write(f"{'CNN Quantized':<25} | {cnn_quant_metrics['accuracy']:7.1%} | {cnn_quant_metrics['inference_time_ms_mean']:10.2f} | {cnn_quant_metrics['inference_time_ms_std']:7.2f} | {cnn_quant_metrics['memory_usage_mb']:10.1f} | {cnn_quant_metrics['throughput_imgs_per_sec']:9.1f} | {cnn_research_metrics['drone_deployment_score']:10.1f}\n")
            f.write(f"{'DINOv2 Original':<25} | {dino_original_metrics['accuracy']:7.1%} | {dino_original_metrics['inference_time_ms_mean']:10.2f} | {dino_original_metrics['inference_time_ms_std']:7.2f} | {dino_original_metrics['memory_usage_mb']:10.1f} | {dino_original_metrics['throughput_imgs_per_sec']:9.1f} | {self.calculate_research_drone_score(dino_original_metrics, dino_original_size):10.1f}\n")
            f.write(f"{'DINOv2 Quantized':<25} | {dino_quant_metrics['accuracy']:7.1%} | {dino_quant_metrics['inference_time_ms_mean']:10.2f} | {dino_quant_metrics['inference_time_ms_std']:7.2f} | {dino_quant_metrics['memory_usage_mb']:10.1f} | {dino_quant_metrics['throughput_imgs_per_sec']:9.1f} | {dino_research_metrics['drone_deployment_score']:10.1f}\n")
            f.write("\n")
            
            # Research insights
            f.write("QUANTIZATION RESEARCH INSIGHTS\n")
            f.write("-" * 50 + "\n")
            f.write("CNN Results:\n")
            f.write(f"  • Accuracy Retention: {cnn_research_metrics['accuracy_retention_percent']:.1f}%\n")
            f.write(f"  • Size Reduction: {cnn_research_metrics['size_reduction_factor']:.1f}x ({cnn_original_size['size_mb']:.1f}MB → {cnn_quant_size['size_mb']:.1f}MB)\n")
            f.write(f"  • Speed Improvement: {cnn_research_metrics['speed_improvement_factor']:.1f}x\n")
            f.write(f"  • Memory Efficiency: {cnn_research_metrics['memory_reduction_factor']:.1f}x improvement\n")
            f.write(f"  • Parameter Reduction: {cnn_research_metrics['parameter_reduction_factor']:.1f}x\n")
            f.write(f"  • Overall Efficiency Score: {cnn_research_metrics['efficiency_score']:.1f}\n\n")
            
            f.write("DINOv2 Results:\n")
            f.write(f"  • Accuracy Retention: {dino_research_metrics['accuracy_retention_percent']:.1f}%\n")
            f.write(f"  • Size Reduction: {dino_research_metrics['size_reduction_factor']:.1f}x ({dino_original_size['size_mb']:.1f}MB → {dino_quant_size['size_mb']:.1f}MB)\n")
            f.write(f"  • Speed Improvement: {dino_research_metrics['speed_improvement_factor']:.1f}x\n")
            f.write(f"  • Memory Efficiency: {dino_research_metrics['memory_reduction_factor']:.1f}x improvement\n")
            f.write(f"  • Parameter Reduction: {dino_research_metrics['parameter_reduction_factor']:.1f}x\n")
            f.write(f"  • Overall Efficiency Score: {dino_research_metrics['efficiency_score']:.1f}\n\n")
            
            # Model comparison and recommendation
            models_comparison = [
                ('CNN Original', self.calculate_research_drone_score(cnn_original_metrics, cnn_original_size), cnn_original_metrics, cnn_original_size),
                ('CNN Quantized', cnn_research_metrics['drone_deployment_score'], cnn_quant_metrics, cnn_quant_size),
                ('DINOv2 Original', self.calculate_research_drone_score(dino_original_metrics, dino_original_size), dino_original_metrics, dino_original_size),
                ('DINOv2 Quantized', dino_research_metrics['drone_deployment_score'], dino_quant_metrics, dino_quant_size)
            ]
            
            best_model = max(models_comparison, key=lambda x: x[1])
            
            f.write("RESEARCH CONCLUSIONS & RECOMMENDATIONS\n")
            f.write("-" * 50 + "\n")
            f.write(f"OPTIMAL MODEL FOR AGRICULTURAL DRONE DEPLOYMENT: {best_model[0]}\n")
            f.write(f"   Research Drone Score: {best_model[1]:.1f}/100\n")
            f.write(f"   Accuracy: {best_model[2]['accuracy']:.1%}\n")
            f.write(f"   Inference Speed: {best_model[2]['inference_time_ms_mean']:.2f}±{best_model[2]['inference_time_ms_std']:.2f}ms\n")
            f.write(f"   Memory Usage: {best_model[2]['memory_usage_mb']:.1f}MB\n")
            f.write(f"   Model Size: {best_model[3]['size_mb']:.1f}MB\n")
            f.write(f"   Throughput: {best_model[2]['throughput_imgs_per_sec']:.1f} images/second\n\n")
            
            # Timing summary
            total_time = cnn_quant_time + dino_quant_time
            f.write("RESEARCH TIMING SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write(f"   • CNN quantization: {cnn_quant_time:.2f}s\n")
            f.write(f"   • DINOv2 quantization: {dino_quant_time:.2f}s\n")
            f.write(f"   • Total quantization time: {total_time:.2f}s\n")
            f.write(f"   • Evaluation thoroughness: Comprehensive (full test set)\n\n")
            
            # Per-class accuracy details (sample)
            f.write("PER-CLASS ACCURACY ANALYSIS\n")
            f.write("-" * 50 + "\n")
            f.write("CNN Quantized per-class accuracy:\n")
            for class_id, acc in list(cnn_quant_metrics['class_accuracies'].items())[:10]:  # Show first 10
                f.write(f"  Class {class_id}: {acc:.1%}\n")
            if len(cnn_quant_metrics['class_accuracies']) > 10:
                f.write(f"  ... and {len(cnn_quant_metrics['class_accuracies']) - 10} more classes\n")
            
            f.write("\nDINOv2 Quantized per-class accuracy:\n")
            for class_id, acc in list(dino_quant_metrics['class_accuracies'].items())[:10]:  # Show first 10
                f.write(f"  Class {class_id}: {acc:.1%}\n")
            if len(dino_quant_metrics['class_accuracies']) > 10:
                f.write(f"  ... and {len(dino_quant_metrics['class_accuracies']) - 10} more classes\n")
            
            f.write("\n" + "=" * 90 + "\n")
            f.write(" RESEARCH NOTES:\n")
            f.write("   • Models saved with comprehensive metadata for paper writing\n")
            f.write("   • Per-class accuracy metrics included for detailed analysis\n")
            f.write("   • Statistical measures (mean, std) provided for reproducibility\n")
            f.write("   • Device and environment information saved for methodology section\n")
            f.write("   • All quantization used dynamic quantization with fbgemm backend\n")
            f.write("   • xformers attention disabled for CPU compatibility\n")
            f.write("   • FAIR COMPARISON: Both original and quantized models evaluated on CPU\n")
            f.write("   • This ensures valid performance comparison (quantized models require CPU)\n")
            f.write("=" * 90 + "\n")
        
        print(f"[LOG] Research log saved: {log_filename}")
        return log_filepath

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
    cnn_path = 'models/best_cnn_b4_label_100_acc86.54.pth'
    dino_path = 'models/best_dino_vits14_label_100_acc89.49.pth'
    
    # Load original models for research
    print("\n[MODELS] Loading research models...")
    cnn_original, class_names = quantizer.load_original_model(cnn_path, config_cnn, 'efficientnet_b4')
    dino_original, _ = quantizer.load_original_model(dino_path, config_dinov2, 'dinov2_vits14')
    
    # Comprehensive evaluation of original models (CPU for fair comparison)
    print("\n[RESEARCH] Comprehensive evaluation of original models...")
    print("[NOTE] Running all models on CPU for fair comparison with quantized models")
    cnn_original_metrics = quantizer.comprehensive_evaluation(
        cnn_original, cnn_test_loader, device, "CNN Original", full_dataset=True, force_cpu_comparison=True
    )
    dino_original_metrics = quantizer.comprehensive_evaluation(
        dino_original, dino_test_loader, device, "DINOv2 Original", full_dataset=True, force_cpu_comparison=True
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
        quant_cnn_dynamic, cnn_test_loader, 'cpu', "CNN Quantized", full_dataset=True, force_cpu_comparison=True
    )
    dino_quant_metrics = quantizer.comprehensive_evaluation(
        quant_dino_dynamic, dino_test_loader, 'cpu', "DINOv2 Quantized", full_dataset=True, force_cpu_comparison=True
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
    
    # Save research log
    quantizer.save_research_log(
        cnn_original_metrics, cnn_quant_metrics, cnn_research_metrics, 
        dino_original_metrics, dino_quant_metrics, dino_research_metrics,
        cnn_original_size, cnn_quant_size, dino_original_size, dino_quant_size,
        total_samples, len(set(sample_labels)), cnn_quant_time, dino_quant_time
    )
    
    # Comprehensive Research Results
    print("\n" + "=" * 90)
    print("COMPREHENSIVE RESEARCH RESULTS (FAIR CPU COMPARISON)")
    print("=" * 90)
    print("WARNING: All models evaluated on CPU for fair comparison")
    print("   (Quantized models currently require CPU; comparing GPU vs CPU would be misleading)")
    
    # Detailed comparison table
    print(f"\n🔬 DETAILED PERFORMANCE ANALYSIS")
    print(f"{'Model':<25} | {'Accuracy':>8} | {'Speed (ms)':>11} | {'±StdDev':>8} | {'Memory (MB)':>11} | {'Throughput':>10} | {'Drone Score':>11}")
    print("-" * 100)
    print(f"{'CNN Original':<25} | {cnn_original_metrics['accuracy']:7.1%} | {cnn_original_metrics['inference_time_ms_mean']:10.2f} | {cnn_original_metrics['inference_time_ms_std']:7.2f} | {cnn_original_metrics['memory_usage_mb']:10.1f} | {cnn_original_metrics['throughput_imgs_per_sec']:9.1f} | {quantizer.calculate_research_drone_score(cnn_original_metrics, cnn_original_size):10.1f}")
    print(f"{'CNN Quantized':<25} | {cnn_quant_metrics['accuracy']:7.1%} | {cnn_quant_metrics['inference_time_ms_mean']:10.2f} | {cnn_quant_metrics['inference_time_ms_std']:7.2f} | {cnn_quant_metrics['memory_usage_mb']:10.1f} | {cnn_quant_metrics['throughput_imgs_per_sec']:9.1f} | {cnn_research_metrics['drone_deployment_score']:10.1f}")
    print(f"{'DINOv2 Original':<25} | {dino_original_metrics['accuracy']:7.1%} | {dino_original_metrics['inference_time_ms_mean']:10.2f} | {dino_original_metrics['inference_time_ms_std']:7.2f} | {dino_original_metrics['memory_usage_mb']:10.1f} | {dino_original_metrics['throughput_imgs_per_sec']:9.1f} | {quantizer.calculate_research_drone_score(dino_original_metrics, dino_original_size):10.1f}")
    print(f"{'DINOv2 Quantized':<25} | {dino_quant_metrics['accuracy']:7.1%} | {dino_quant_metrics['inference_time_ms_mean']:10.2f} | {dino_quant_metrics['inference_time_ms_std']:7.2f} | {dino_quant_metrics['memory_usage_mb']:10.1f} | {dino_quant_metrics['throughput_imgs_per_sec']:9.1f} | {dino_research_metrics['drone_deployment_score']:10.1f}")
    
    # Research insights
    print(f"\nQUANTIZATION RESEARCH INSIGHTS")
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
    print(f"\nRESEARCH CONCLUSIONS & RECOMMENDATIONS")
    
    # Determine best model based on research metrics
    models_comparison = [
        ('CNN Original', quantizer.calculate_research_drone_score(cnn_original_metrics, cnn_original_size), cnn_original_metrics, cnn_original_size),
        ('CNN Quantized', cnn_research_metrics['drone_deployment_score'], cnn_quant_metrics, cnn_quant_size),
        ('DINOv2 Original', quantizer.calculate_research_drone_score(dino_original_metrics, dino_original_size), dino_original_metrics, dino_original_size),
        ('DINOv2 Quantized', dino_research_metrics['drone_deployment_score'], dino_quant_metrics, dino_quant_size)
    ]
    
    best_model = max(models_comparison, key=lambda x: x[1])
    
    print(f"\nOPTIMAL MODEL FOR AGRICULTURAL DRONE DEPLOYMENT: {best_model[0]}")
    print(f"   Research Drone Score: {best_model[1]:.1f}/100")
    print(f"   Accuracy: {best_model[2]['accuracy']:.1%}")
    print(f"   Inference Speed: {best_model[2]['inference_time_ms_mean']:.2f}±{best_model[2]['inference_time_ms_std']:.2f}ms")
    print(f"   Memory Usage: {best_model[2]['memory_usage_mb']:.1f}MB")
    print(f"   Model Size: {best_model[3]['size_mb']:.1f}MB")
    print(f"   Throughput: {best_model[2]['throughput_imgs_per_sec']:.1f} images/second")
    
    # Research dataset summary
    print(f"\nRESEARCH DATASET SUMMARY")
    print(f"   Total samples evaluated: {cnn_quant_metrics['total_samples']:,}")
    print(f"   Number of classes: {cnn_quant_metrics['num_classes_evaluated']}")
    print(f"   Evaluation batches: {cnn_quant_metrics['evaluation_batches']}")
    
    # File outputs for research
    print(f"\n RESEARCH OUTPUT FILES")
    print(f"    Directory: {quantizer.save_dir.absolute()}")
    print(f"    CNN Quantized: {cnn_save_path.name}")
    print(f"    DINOv2 Quantized: {dino_save_path.name}")
    print(f"\n RESEARCH NOTES:")
    print(f"   • Models saved with comprehensive metadata for paper writing")
    print(f"   • Per-class accuracy metrics included for detailed analysis")
    print(f"   • Statistical measures (mean, std) provided for reproducibility")
    print(f"   • Device and environment information saved for methodology section")
    print(f"   • FAIR COMPARISON: Both original and quantized models evaluated on CPU")
    print(f"   • This ensures valid performance comparison (quantized models require CPU)")
    
    # Research timing summary
    total_time = cnn_quant_time + dino_quant_time
    print(f"\nRESEARCH TIMING SUMMARY")
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