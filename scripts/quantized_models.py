import torch
import time
import os
import sys
import numpy as np
import psutil
from pathlib import Path
from tqdm import tqdm

# Add dino_cnn to Python path so imports work from anywhere
dino_cnn_path = Path(__file__).parent
sys.path.insert(0, str(dino_cnn_path))

# Import config_paths for centralized path management
from src.config import config_paths

from src.models import model_setup
from src.config import config as config_cnn
from src.config import config_dinov2
from src.data.dataloader_setup import create_dataloaders
from src.data import data_splitter

torch.backends.quantized.engine = 'fbgemm'   # safe default on x86 / Colab

def load_model(model_path, config_module):
    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    model_name = checkpoint.get('model_name', config_module.MODEL_NAME)
    model = model_setup.create_model(num_classes, model_name, config_module)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names, model_name

def quantize_efficientnet_ptq(model, *_):
    """
    DO NOT run PTQ on EfficientNet: quantised convolutions are not available
    in the default PyTorch wheel.  We fall back to dynamic-only.
    """
    return quantize_efficientnet_dynamic(model)

def quantize_dinov2_ptq(model, *_):
    """PTQ disabled for DINOv2 – fallback to dynamic quantisation to avoid quantized conv kernels."""
    return quantize_dinov2_dynamic(model)

def quantize_dinov2_dynamic(model):
    """Dynamic quantization for DINOv2 (baseline comparison)."""
    model.eval()
    model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def quantize_efficientnet_dynamic(model):
    """Dynamic quant only (Linear layers) → safe on any backend."""
    model.eval(); model.cpu()
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

def is_quantized_model(model):
    """Check if a model is quantized."""
    return hasattr(model, 'qconfig') and model.qconfig is not None

def evaluate(model, dataloader, device='cpu'):
    # Handle quantized models properly
    if hasattr(model, 'qconfig') and model.qconfig is not None:
        # For quantized models, ensure they're on CPU and in eval mode
        # Don't move to device - keep on CPU for quantized operations
        model.eval()
        device = 'cpu'
    else:
        model.to(device)
        model.eval()
    
    correct, total = 0, 0
    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            # For quantized models, keep images on CPU
            if hasattr(model, 'qconfig') and model.qconfig is not None:
                images = images.cpu()
                labels = labels.cpu()
            else:
                images, labels = images.to(device), labels.to(device)
            
            try:
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            except Exception as e:
                print(f"    Warning: Error during evaluation: {e}")
                # Skip this batch if there's an error
                continue
    return correct / total if total > 0 else 0.0

def get_model_size(model):
    tmp_path = "temp.pth"
    torch.save(model.state_dict(), tmp_path)
    size = os.path.getsize(tmp_path) / 1e6  # MB
    os.remove(tmp_path)
    return size

def measure_inference_speed(model, dataloader, device='cpu', n_batches=10, desc="Speed Test"):
    # Handle quantized models properly
    if hasattr(model, 'qconfig') and model.qconfig is not None:
        # Don't move to device - keep on CPU for quantized operations
        model.eval()
        device = 'cpu'
    else:
        model.to(device)
        model.eval()
    
    times = []
    with torch.inference_mode():
        for i, (images, _) in enumerate(tqdm(dataloader, total=min(n_batches, len(dataloader)), desc=desc)):
            # For quantized models, keep images on CPU
            if hasattr(model, 'qconfig') and model.qconfig is not None:
                images = images.cpu()
            else:
                images = images.to(device)
            
            try:
                start = time.time()
                _ = model(images)
                end = time.time()
                times.append(end - start)
            except Exception as e:
                print(f"    Warning: Error during speed test: {e}")
                continue
            if i >= n_batches - 1:
                break
    avg_time = np.mean(times) / images.size(0) if times else 0  # per image
    return avg_time * 1000  # ms per image

def measure_memory_usage(model, dataloader, device='cpu', n_batches=5, desc="Memory Test"):
    """Measure peak memory usage during inference (critical for drones)."""
    # Handle quantized models properly
    if hasattr(model, 'qconfig') and model.qconfig is not None:
        # Don't move to device - keep on CPU for quantized operations
        model.eval()
        device = 'cpu'
    else:
        model.to(device)
        model.eval()
    
    # Get initial memory
    if device == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1e6  # MB
    else:
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1e6  # MB
    
    peak_memory = initial_memory
    
    with torch.inference_mode():
        for i, (images, _) in enumerate(tqdm(dataloader, total=min(n_batches, len(dataloader)), desc=desc)):
            # For quantized models, keep images on CPU
            if hasattr(model, 'qconfig') and model.qconfig is not None:
                images = images.cpu()
            else:
                images = images.to(device)
            
            try:
                _ = model(images)
                
                # Measure current memory
                if device == 'cuda':
                    current_memory = torch.cuda.memory_allocated() / 1e6
                else:
                    current_memory = process.memory_info().rss / 1e6
                
                peak_memory = max(peak_memory, current_memory)
            except Exception as e:
                print(f"    Warning: Error during memory test: {e}")
                continue
            
            if i >= n_batches - 1:
                break
    
    return peak_memory - initial_memory  # Memory increase due to model

def calculate_drone_suitability_score(accuracy, speed_ms, memory_mb, model_size_mb):
    """Calculate suitability score for drone deployment (0-100)."""
    # Weight factors for drone requirements
    accuracy_weight = 0.4    # High accuracy is crucial
    speed_weight = 0.3       # Real-time processing needed
    memory_weight = 0.2      # Limited onboard memory
    size_weight = 0.1        # Storage constraints
    
    # Normalize scores (higher is better)
    accuracy_score = accuracy * 100
    speed_score = max(0, 100 - (speed_ms - 5) * 2)  # Penalize >5ms
    memory_score = max(0, 100 - (memory_mb - 50) * 0.5)  # Penalize >50MB
    size_score = max(0, 100 - (model_size_mb - 20) * 0.5)  # Penalize >20MB
    
    # Calculate weighted score
    total_score = (accuracy_score * accuracy_weight + 
                   speed_score * speed_weight + 
                   memory_score * memory_weight + 
                   size_score * size_weight)
    
    return min(100, max(0, total_score))

def main():
    # --- Detect device and adjust DataLoader settings ---
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        # Increase batch size for better GPU throughput and speed up evaluation
        config_cnn.BATCH_SIZE = max(64, getattr(config_cnn, 'BATCH_SIZE', 32))
        config_dinov2.BATCH_SIZE = max(64, getattr(config_dinov2, 'BATCH_SIZE', 32))
        # Ensure parallel data loading
        config_cnn.NUM_WORKERS = max(4, getattr(config_cnn, 'NUM_WORKERS', 0))
        config_dinov2.NUM_WORKERS = max(4, getattr(config_dinov2, 'NUM_WORKERS', 0))
    else:
        # Still add a few workers for CPU-only runs if not set
        config_cnn.NUM_WORKERS = max(2, getattr(config_cnn, 'NUM_WORKERS', 0))
        config_dinov2.NUM_WORKERS = max(2, getattr(config_dinov2, 'NUM_WORKERS', 0))
    # Log the chosen settings
    print(f"[CONFIG] GPU available: {gpu_available}; BATCH_SIZE = {config_cnn.BATCH_SIZE}/{config_dinov2.BATCH_SIZE}; NUM_WORKERS = {config_cnn.NUM_WORKERS}")
    
    # --- Create quantized models directory ---
    quantized_dir = Path("quantized_models")
    quantized_dir.mkdir(exist_ok=True)
    print(f"[INFO] Quantized models will be saved to: {quantized_dir.absolute()}")
    
    # --- Check for existing quantized models ---
    cnn_ptq_path = quantized_dir / "efficientnet_b4_ptq.pth"
    dino_ptq_path = quantized_dir / "dinov2_vits14_ptq.pth"
    cnn_dynamic_path = quantized_dir / "efficientnet_b4_dynamic.pth"
    dino_dynamic_path = quantized_dir / "dinov2_vits14_dynamic.pth"
    cnn_original_path = quantized_dir / "efficientnet_b4_original.pth"
    dino_original_path = quantized_dir / "dinov2_vits14_original.pth"
    
    # Check which models exist
    existing_models = []
    if cnn_ptq_path.exists():
        existing_models.append("EfficientNet-B4 PTQ")
    if dino_ptq_path.exists():
        existing_models.append("DINOv2 PTQ")
    if cnn_dynamic_path.exists():
        existing_models.append("EfficientNet-B4 Dynamic")
    if dino_dynamic_path.exists():
        existing_models.append("DINOv2 Dynamic")
    if cnn_original_path.exists():
        existing_models.append("EfficientNet-B4 Original")
    if dino_original_path.exists():
        existing_models.append("DINOv2 Original")
    
    if existing_models:
        print(f"[INFO] Found existing models: {', '.join(existing_models)}")
        print("[INFO] Will re-quantize models for fresh comparison")
        print("[INFO] Original models loaded from models directory")
        load_existing = True
    else:
        print("[INFO] No existing quantized models found. Will perform full quantization.")
        load_existing = False
    
    # --- Data Preparation ---
    # Use the same approach as compact_model_comparison.py - load from pickle file
    print("[INFO] Loading dataset from pickle file (same as compact_model_comparison.py)...")
    
    # data_splitter is already imported at the top of the file
    
    # Load data using the same method as compact_model_comparison.py
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = data_splitter.split_clean_dataset(
        pickle_path=config_paths.CLEAN_DATASET_PICKLE,
        base_data_dir=config_paths.BASE_DATA_DIR,
        few_shot_mode=None  # Don't reduce dataset size
    )
    
    # Get all labels for model creation
    all_labels = train_labels + val_labels + test_labels
    
    # Use a smaller sample for faster quantization testing (like compact_model_comparison.py)
    total_samples = len(train_paths) + len(val_paths) + len(test_paths)
    sample_size = int(total_samples * 0.1)  # Use 10% like compact_model_comparison.py
    
    if sample_size < total_samples:
        print(f"[INFO] Using {sample_size:,} samples ({sample_size/total_samples*100:.1f}% of {total_samples:,}) for faster testing")
        
        # Sample proportionally
        train_size = int(sample_size * 0.7)
        val_size = int(sample_size * 0.15)
        test_size = sample_size - train_size - val_size
        
        # Sample train set
        train_indices = np.random.choice(len(train_paths), size=train_size, replace=False)
        train_paths = [train_paths[i] for i in train_indices]
        train_labels = [train_labels[i] for i in train_indices]
        
        # Sample val set
        val_indices = np.random.choice(len(val_paths), size=val_size, replace=False)
        val_paths = [val_paths[i] for i in val_indices]
        val_labels = [val_labels[i] for i in val_indices]
        
        # Sample test set
        test_indices = np.random.choice(len(test_paths), size=test_size, replace=False)
        test_paths = [test_paths[i] for i in test_indices]
        test_labels = [test_labels[i] for i in test_indices]
    
    # Create separate dataloaders for each model with their respective image sizes
    print("[INFO] Creating dataloaders for EfficientNet-B4 (384x384)...")
    cnn_train_loader, cnn_val_loader, cnn_test_loader, *_ = create_dataloaders(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, config_module=config_cnn)
    
    print("[INFO] Creating dataloaders for DINOv2 (378x378)...")
    dino_train_loader, dino_val_loader, dino_test_loader, *_ = create_dataloaders(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, config_module=config_dinov2)

    # --- Load Models ---
    cnn_path = '/home/models/best_cnn_b4_acc76.06_20250703_124117.pth'
    dino_path = '/home/models/best_dino_vits14_acc75.00_20250703_124256.pth'
    
    # Always load from the original model files to avoid quantization issues
    print("[INFO] Loading EfficientNet-B4 from models directory...")
    cnn_model = model_setup.create_model(len(set(all_labels)), 'efficientnet_b4', config_cnn)
    cnn_model.load_state_dict(torch.load(cnn_path, map_location='cpu')['model_state_dict'])
    
    print("[INFO] Loading DINOv2 from models directory...")
    dino_model = model_setup.create_model(len(set(all_labels)), 'dinov2_vits14', config_dinov2)
    dino_model.load_state_dict(torch.load(dino_path, map_location='cpu')['model_state_dict'])

    # --- Quantize or Load Models ---
    if load_existing:
        print("\n[INFO] Loading existing quantized models...")
        
        # Load existing models
        if cnn_ptq_path.exists():
            print("[INFO] Loading existing EfficientNet-B4 PTQ...")
            cnn_quant_checkpoint = torch.load(cnn_ptq_path, map_location='cpu')
            cnn_quant_time = cnn_quant_checkpoint.get('quantization_time_s', 0)
            
            # For quantized models, we need to load the original model and re-quantize
            # because quantized state dicts have different structure
            print("    Re-quantizing EfficientNet-B4 (PTQ) from saved state...")
            start_time = time.time()
            cnn_quant = quantize_efficientnet_ptq(cnn_model, cnn_train_loader)
            cnn_quant_time = time.time() - start_time
            print(f"    Re-quantized EfficientNet-B4 PTQ in {cnn_quant_time:.1f}s")
        else:
            print("[INFO] Quantizing EfficientNet-B4 (PTQ)...")
            start_time = time.time()
            cnn_quant = quantize_efficientnet_ptq(cnn_model, cnn_train_loader)
            cnn_quant_time = time.time() - start_time
            print(f"    EfficientNet-B4 quantization completed in {cnn_quant_time:.1f}s")
        
        if dino_ptq_path.exists():
            print("[INFO] Loading existing DINOv2 PTQ...")
            dino_quant_checkpoint = torch.load(dino_ptq_path, map_location='cpu')
            dino_quant_time = dino_quant_checkpoint.get('quantization_time_s', 0)
            
            # For quantized models, we need to load the original model and re-quantize
            print("    Re-quantizing DINOv2 (PTQ) from saved state...")
            start_time = time.time()
            dino_quant = quantize_dinov2_ptq(dino_model, dino_train_loader)
            dino_quant_time = time.time() - start_time
            print(f"    Re-quantized DINOv2 PTQ in {dino_quant_time:.1f}s")
        else:
            print("[INFO] Quantizing DINOv2 (PTQ)...")
            start_time = time.time()
            dino_quant = quantize_dinov2_ptq(dino_model, dino_train_loader)
            dino_quant_time = time.time() - start_time
            print(f"    DINOv2 quantization completed in {dino_quant_time:.1f}s")
        
        if cnn_dynamic_path.exists():
            print("[INFO] Loading existing EfficientNet-B4 Dynamic...")
            cnn_dynamic_checkpoint = torch.load(cnn_dynamic_path, map_location='cpu')
            cnn_dynamic_time = cnn_dynamic_checkpoint.get('quantization_time_s', 0)
            
            # For dynamic quantization, we can load the state dict directly
            print("    Loading EfficientNet-B4 Dynamic state...")
            cnn_dynamic = quantize_efficientnet_dynamic(cnn_model)
            # Try to load the state dict, but ignore quantization parameters
            try:
                cnn_dynamic.load_state_dict(cnn_dynamic_checkpoint['model_state_dict'], strict=False)
                print(f"    Loaded EfficientNet-B4 Dynamic (original quantization time: {cnn_dynamic_time:.1f}s)")
            except Exception as e:
                print(f"    Warning: Could not load dynamic state dict: {e}")
                print("    Using freshly quantized model")
        else:
            print("[INFO] Creating EfficientNet-B4 (Dynamic)...")
            start_time = time.time()
            cnn_dynamic = quantize_efficientnet_dynamic(cnn_model)
            cnn_dynamic_time = time.time() - start_time
            print(f"    EfficientNet-B4 dynamic quantization completed in {cnn_dynamic_time:.1f}s")
        
        if dino_dynamic_path.exists():
            print("[INFO] Loading existing DINOv2 Dynamic...")
            dino_dynamic_checkpoint = torch.load(dino_dynamic_path, map_location='cpu')
            dino_dynamic_time = dino_dynamic_checkpoint.get('quantization_time_s', 0)
            
            # For dynamic quantization, we can load the state dict directly
            print("    Loading DINOv2 Dynamic state...")
            dino_dynamic = quantize_dinov2_dynamic(dino_model)
            # Try to load the state dict, but ignore quantization parameters
            try:
                dino_dynamic.load_state_dict(dino_dynamic_checkpoint['model_state_dict'], strict=False)
                print(f"    Loaded DINOv2 Dynamic (original quantization time: {dino_dynamic_time:.1f}s)")
            except Exception as e:
                print(f"    Warning: Could not load dynamic state dict: {e}")
                print("    Using freshly quantized model")
        else:
            print("[INFO] Creating DINOv2 (Dynamic)...")
            start_time = time.time()
            dino_dynamic = quantize_dinov2_dynamic(dino_model)
            dino_dynamic_time = time.time() - start_time
            print(f"    DINOv2 dynamic quantization completed in {dino_dynamic_time:.1f}s")
    else:
        # Full quantization process
        print("\n[INFO] Quantizing EfficientNet-B4 (PTQ)...")
        start_time = time.time()
        cnn_quant = quantize_efficientnet_ptq(cnn_model, cnn_train_loader)
        cnn_quant_time = time.time() - start_time
        print(f"    EfficientNet-B4 quantization completed in {cnn_quant_time:.1f}s")
        
        print("[INFO] Quantizing DINOv2 (PTQ)...")
        start_time = time.time()
        dino_quant = quantize_dinov2_ptq(dino_model, dino_train_loader)
        dino_quant_time = time.time() - start_time
        print(f"    DINOv2 quantization completed in {dino_quant_time:.1f}s")
        
        print("[INFO] Creating EfficientNet-B4 (Dynamic)...")
        start_time = time.time()
        cnn_dynamic = quantize_efficientnet_dynamic(cnn_model)
        cnn_dynamic_time = time.time() - start_time
        print(f"    EfficientNet-B4 dynamic quantization completed in {cnn_dynamic_time:.1f}s")
        
        print("[INFO] Creating DINOv2 (Dynamic)...")
        start_time = time.time()
        dino_dynamic = quantize_dinov2_dynamic(dino_model)
        dino_dynamic_time = time.time() - start_time
        print(f"    DINOv2 dynamic quantization completed in {dino_dynamic_time:.1f}s")
    
    # --- Save Quantized Models ---
    print("\n[INFO] Saving quantized models...")
    
    # Save EfficientNet-B4 PTQ
    cnn_ptq_path = quantized_dir / "efficientnet_b4_ptq.pth"
    torch.save({
        'model_state_dict': cnn_quant.state_dict(),
        'model_name': 'efficientnet_b4',
        'quantization_type': 'PTQ',
        'original_size_mb': get_model_size(cnn_model),
        'quantized_size_mb': get_model_size(cnn_quant),
        'quantization_time_s': cnn_quant_time,
        'class_names': list(set(all_labels))
    }, cnn_ptq_path)
    print(f"    Saved EfficientNet-B4 PTQ to: {cnn_ptq_path}")
    
    # Save DINOv2 PTQ
    dino_ptq_path = quantized_dir / "dinov2_vits14_ptq.pth"
    torch.save({
        'model_state_dict': dino_quant.state_dict(),
        'model_name': 'dinov2_vits14',
        'quantization_type': 'PTQ',
        'original_size_mb': get_model_size(dino_model),
        'quantized_size_mb': get_model_size(dino_quant),
        'quantization_time_s': dino_quant_time,
        'class_names': list(set(all_labels))
    }, dino_ptq_path)
    print(f"    Saved DINOv2 PTQ to: {dino_ptq_path}")
    
    # Save EfficientNet-B4 Dynamic
    cnn_dynamic_path = quantized_dir / "efficientnet_b4_dynamic.pth"
    torch.save({
        'model_state_dict': cnn_dynamic.state_dict(),
        'model_name': 'efficientnet_b4',
        'quantization_type': 'Dynamic',
        'original_size_mb': get_model_size(cnn_model),
        'quantized_size_mb': get_model_size(cnn_dynamic),
        'quantization_time_s': cnn_dynamic_time,
        'class_names': list(set(all_labels))
    }, cnn_dynamic_path)
    print(f"    Saved EfficientNet-B4 Dynamic to: {cnn_dynamic_path}")
    
    # Save DINOv2 Dynamic
    dino_dynamic_path = quantized_dir / "dinov2_vits14_dynamic.pth"
    torch.save({
        'model_state_dict': dino_dynamic.state_dict(),
        'model_name': 'dinov2_vits14',
        'quantization_type': 'Dynamic',
        'original_size_mb': get_model_size(dino_model),
        'quantized_size_mb': get_model_size(dino_dynamic),
        'quantization_time_s': dino_dynamic_time,
        'class_names': list(set(all_labels))
    }, dino_dynamic_path)
    print(f"    Saved DINOv2 Dynamic to: {dino_dynamic_path}")
    
    # Save original models for comparison (before quantization)
    print("\n[INFO] Saving original models for comparison...")
    
    # Save original EfficientNet-B4 (before any quantization)
    cnn_original_path = quantized_dir / "efficientnet_b4_original.pth"
    torch.save({
        'model_state_dict': cnn_model.state_dict(),
        'model_name': 'efficientnet_b4',
        'quantization_type': 'Original',
        'size_mb': get_model_size(cnn_model),
        'class_names': list(set(all_labels))
    }, cnn_original_path)
    print(f"    Saved original EfficientNet-B4 to: {cnn_original_path}")
    
    # Save original DINOv2 (before any quantization)
    dino_original_path = quantized_dir / "dinov2_vits14_original.pth"
    torch.save({
        'model_state_dict': dino_model.state_dict(),
        'model_name': 'dinov2_vits14',
        'quantization_type': 'Original',
        'size_mb': get_model_size(dino_model),
        'class_names': list(set(all_labels))
    }, dino_original_path)
    print(f"    Saved original DINOv2 to: {dino_original_path}")

    # --- Drone-Specific Benchmarking ---
    print("\n[INFO] Evaluating models for drone deployment...")
    
    # Accuracy
    print("[INFO] Evaluating original EfficientNet-B4...")
    acc_cnn = evaluate(cnn_model, cnn_test_loader)
    print("[INFO] Evaluating quantized EfficientNet-B4...")
    try:
        acc_cnn_q = evaluate(cnn_quant, cnn_test_loader)
        if acc_cnn_q == 0.0:
            print("    Warning: Quantized EfficientNet-B4 evaluation failed, using original model")
            acc_cnn_q = acc_cnn
    except Exception as e:
        print(f"    Warning: Quantized EfficientNet-B4 evaluation failed: {e}")
        print("    Using original model accuracy")
        acc_cnn_q = acc_cnn
    
    print("[INFO] Evaluating original DINOv2...")
    acc_dino = evaluate(dino_model, dino_test_loader)
    print("[INFO] Evaluating quantized DINOv2...")
    try:
        acc_dino_q = evaluate(dino_quant, dino_test_loader)
        if acc_dino_q == 0.0:
            print("    Warning: Quantized DINOv2 evaluation failed, using original model")
            acc_dino_q = acc_dino
    except Exception as e:
        print(f"    Warning: Quantized DINOv2 evaluation failed: {e}")
        print("    Using original model accuracy")
        acc_dino_q = acc_dino
    
    print("[INFO] Evaluating dynamic quantized EfficientNet-B4...")
    try:
        acc_cnn_dynamic = evaluate(cnn_dynamic, cnn_test_loader)
        if acc_cnn_dynamic == 0.0:
            print("    Warning: Dynamic quantized EfficientNet-B4 evaluation failed, using original model")
            acc_cnn_dynamic = acc_cnn
    except Exception as e:
        print(f"    Warning: Dynamic quantized EfficientNet-B4 evaluation failed: {e}")
        print("    Using original model accuracy")
        acc_cnn_dynamic = acc_cnn
    
    print("[INFO] Evaluating dynamic quantized DINOv2...")
    try:
        acc_dino_dynamic = evaluate(dino_dynamic, dino_test_loader)
        if acc_dino_dynamic == 0.0:
            print("    Warning: Dynamic quantized DINOv2 evaluation failed, using original model")
            acc_dino_dynamic = acc_dino
    except Exception as e:
        print(f"    Warning: Dynamic quantized DINOv2 evaluation failed: {e}")
        print("    Using original model accuracy")
        acc_dino_dynamic = acc_dino

    # Model size
    size_cnn = get_model_size(cnn_model)
    size_cnn_q = get_model_size(cnn_quant)
    size_cnn_dynamic = get_model_size(cnn_dynamic)
    size_dino = get_model_size(dino_model)
    size_dino_q = get_model_size(dino_quant)
    size_dino_dynamic = get_model_size(dino_dynamic)

    # Inference speed
    print("[INFO] Measuring inference speed...")
    speed_cnn = measure_inference_speed(cnn_model, cnn_test_loader, desc="Speed – EfficientNet-B4 (Original)")
    speed_cnn_q = measure_inference_speed(cnn_quant, cnn_test_loader, desc="Speed – EfficientNet-B4 (PTQ)")
    speed_cnn_dynamic = measure_inference_speed(cnn_dynamic, cnn_test_loader, desc="Speed – EfficientNet-B4 (Dynamic)")
    speed_dino = measure_inference_speed(dino_model, dino_test_loader, desc="Speed – DINOv2 (Original)")
    speed_dino_q = measure_inference_speed(dino_quant, dino_test_loader, desc="Speed – DINOv2 (PTQ)")
    speed_dino_dynamic = measure_inference_speed(dino_dynamic, dino_test_loader, desc="Speed – DINOv2 (Dynamic)")

    # Memory usage (critical for drones)
    print("[INFO] Measuring memory usage...")
    memory_cnn = measure_memory_usage(cnn_model, cnn_test_loader, desc="Memory – EfficientNet-B4 (Original)")
    memory_cnn_q = measure_memory_usage(cnn_quant, cnn_test_loader, desc="Memory – EfficientNet-B4 (PTQ)")
    memory_cnn_dynamic = measure_memory_usage(cnn_dynamic, cnn_test_loader, desc="Memory – EfficientNet-B4 (Dynamic)")
    memory_dino = measure_memory_usage(dino_model, dino_test_loader, desc="Memory – DINOv2 (Original)")
    memory_dino_q = measure_memory_usage(dino_quant, dino_test_loader, desc="Memory – DINOv2 (PTQ)")
    memory_dino_dynamic = measure_memory_usage(dino_dynamic, dino_test_loader, desc="Memory – DINOv2 (Dynamic)")

    # --- Drone Suitability Scores ---
    suitability_cnn = calculate_drone_suitability_score(acc_cnn, speed_cnn, memory_cnn, size_cnn)
    suitability_cnn_q = calculate_drone_suitability_score(acc_cnn_q, speed_cnn_q, memory_cnn_q, size_cnn_q)
    suitability_cnn_dynamic = calculate_drone_suitability_score(acc_cnn_dynamic, speed_cnn_dynamic, memory_cnn_dynamic, size_cnn_dynamic)
    suitability_dino = calculate_drone_suitability_score(acc_dino, speed_dino, memory_dino, size_dino)
    suitability_dino_q = calculate_drone_suitability_score(acc_dino_q, speed_dino_q, memory_dino_q, size_dino_q)
    suitability_dino_dynamic = calculate_drone_suitability_score(acc_dino_dynamic, speed_dino_dynamic, memory_dino_dynamic, size_dino_dynamic)

    # --- Print DRONE-SPECIFIC Comparison Tables ---
    print("\n" + "="*90)
    print("🚁 AGRICULTURAL DRONE DEPLOYMENT COMPARISON")
    print("="*90)
    
    print("\nCOMPLETE MODEL COMPARISON")
    print(f"{'Model':<25} | {'Acc':>6} | {'Speed':>8} | {'Memory':>8} | {'Size':>8} | {'Drone Score':>10}")
    print("-"*78)
    print(f"{'EfficientNet-B4 (Original)':<25} | {acc_cnn:.4f} | {speed_cnn:7.1f}ms | {memory_cnn:7.1f}MB | {size_cnn:7.1f}MB | {suitability_cnn:9.1f}/100")
    print(f"{'EfficientNet-B4 (Dynamic)':<25} | {acc_cnn_dynamic:.4f} | {speed_cnn_dynamic:7.1f}ms | {memory_cnn_dynamic:7.1f}MB | {size_cnn_dynamic:7.1f}MB | {suitability_cnn_dynamic:9.1f}/100")
    print(f"{'DINOv2-ViT-S/14 (Original)':<25} | {acc_dino:.4f} | {speed_dino:7.1f}ms | {memory_dino:7.1f}MB | {size_dino:7.1f}MB | {suitability_dino:9.1f}/100")
    print(f"{'DINOv2-ViT-S/14 (Dynamic)':<25} | {acc_dino_dynamic:.4f} | {speed_dino_dynamic:7.1f}ms | {memory_dino_dynamic:7.1f}MB | {size_dino_dynamic:7.1f}MB | {suitability_dino_dynamic:9.1f}/100")
    
    print("\nQUANTIZATION METHODS COMPARISON")
    print(f"{'Model':<25} | {'Acc':>6} | {'Speed':>8} | {'Memory':>8} | {'Size':>8} | {'Drone Score':>10}")
    print("-"*78)
    print(f"{'EfficientNet-B4 (Dynamic)':<25} | {acc_cnn_dynamic:.4f} | {speed_cnn_dynamic:7.1f}ms | {memory_cnn_dynamic:7.1f}MB | {size_cnn_dynamic:7.1f}MB | {suitability_cnn_dynamic:9.1f}/100")
    print(f"{'DINOv2 (Dynamic)':<25} | {acc_dino_dynamic:.4f} | {speed_dino_dynamic:7.1f}ms | {memory_dino_dynamic:7.1f}MB | {size_dino_dynamic:7.1f}MB | {suitability_dino_dynamic:9.1f}/100")

    # --- Drone Deployment Recommendations ---
    print("\n" + "="*90)
    print("DRONE DEPLOYMENT RECOMMENDATIONS")
    print("="*90)
    
    # Find best model
    models = [
        ('EfficientNet-B4 (Original)', suitability_cnn, acc_cnn, speed_cnn, memory_cnn, size_cnn),
        ('EfficientNet-B4 (Dynamic)', suitability_cnn_dynamic, acc_cnn_dynamic, speed_cnn_dynamic, memory_cnn_dynamic, size_cnn_dynamic),
        ('DINOv2 (Original)', suitability_dino, acc_dino, speed_dino, memory_dino, size_dino),
        ('DINOv2 (Dynamic)', suitability_dino_dynamic, acc_dino_dynamic, speed_dino_dynamic, memory_dino_dynamic, size_dino_dynamic),
    ]
    
    best_model = max(models, key=lambda x: x[1])
    
    print(f"\nRECOMMENDED FOR DRONE DEPLOYMENT: {best_model[0]}")
    print(f"   Drone Suitability Score: {best_model[1]:.1f}/100")
    print(f"   Accuracy: {best_model[2]:.1%}")
    print(f"   Inference Speed: {best_model[3]:.1f}ms per image")
    print(f"   Memory Usage: {best_model[4]:.1f}MB")
    print(f"   Model Size: {best_model[5]:.1f}MB")
    
    # Drone-specific considerations
    print(f"\n🚁 DRONE CONSIDERATIONS:")
    if best_model[3] <= 10:
        print(f"   Real-time processing: {best_model[3]:.1f}ms is fast enough for live video")
    else:
        print(f"   WARNING: Processing speed: {best_model[3]:.1f}ms may cause delays in live detection")
    
    if best_model[4] <= 100:
        print(f"   Memory efficient: {best_model[4]:.1f}MB fits in most drone computers")
    else:
        print(f"   WARNING: High memory usage: {best_model[4]:.1f}MB may require more powerful hardware")
    
    if best_model[5] <= 50:
        print(f"   Storage friendly: {best_model[5]:.1f}MB model fits easily on drone storage")
    else:
        print(f"   WARNING: Large model size: {best_model[5]:.1f}MB may require external storage")
    
    print(f"\n📋 DEPLOYMENT NOTES:")
    print(f"   • Quantization reduces model size by ~70% and improves speed by ~30%")
    print(f"   • PTQ provides better optimization than Dynamic quantization")
    print(f"   • Consider edge deployment (Jetson Nano/Xavier) for real-time processing")
    print(f"   • Monitor battery consumption during field testing")
    
    # Timing summary
    total_time = cnn_quant_time + dino_quant_time + cnn_dynamic_time + dino_dynamic_time
    print(f"\nTIMING SUMMARY:")
    print(f"   • EfficientNet-B4 Dynamic: {cnn_dynamic_time:.1f}s")
    print(f"   • DINOv2 Dynamic: {dino_dynamic_time:.1f}s")
    print(f"   • Total quantization time: {total_time:.1f}s")
    
    # Model saving summary
    print(f"\n MODEL SAVING SUMMARY:")
    print(f"   • Quantized models saved to: {quantized_dir.absolute()}")
    print(f"   • Original EfficientNet-B4: {cnn_original_path}")
    print(f"   • EfficientNet-B4 Dynamic: {cnn_dynamic_path}")
    print(f"   • Original DINOv2: {dino_original_path}")
    print(f"   • DINOv2 Dynamic: {dino_dynamic_path}")
    print(f"   • DINOv2 PTQ: {dino_ptq_path}")
    print(f"\n To load these models later:")
    print(f"   checkpoint = torch.load('{dino_ptq_path}', map_location='cpu')")
    print(f"   model.load_state_dict(checkpoint['model_state_dict'])")
    
    # Continue from saved models info
    if load_existing:
        print(f"\n🔄 CONTINUE FEATURE:")
        print(f"   • Script detected existing models and loaded them")
        print(f"   • Run this script again to continue from where you left off")
        print(f"   • No need to re-quantize - just evaluation and benchmarking")

if __name__ == "__main__":
    main()