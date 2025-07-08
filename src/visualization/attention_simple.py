#!/usr/bin/env python3
"""
Enhanced attention visualization comparing original vs quantized models.
Shows: Original DINO | Original CNN | Quantized DINO | Quantized CNN
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import sys
import glob
sys.path.append('src')

from src.models.model_setup import create_model

import torch

# Centralized paths
from src.config.config_paths import ATTENTION_VISUALIZATIONS_DIR, ensure_directories, PROJECT_ROOT

# Ensure directories exist
ensure_directories()

class EnhancedAttentionVisualizer:
    """Enhanced attention visualization with original vs quantized model comparison."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}  # Will store loaded models
        self.model_info = {}
        
    def find_models(self):
        """Auto-detect available original and quantized models."""
        models_dir = Path(PROJECT_ROOT) / 'models'
        quantized_dir = Path(PROJECT_ROOT) / 'quantized_models'
        
        model_patterns = {
            'cnn_original': 'best_cnn_*_acc*.pth',
            'dino_original': 'best_dino_*_acc*.pth',
            'cnn_quantized': 'quant_cnn_*.pth',
            'dino_quantized': 'quant_dino_*.pth'
        }
        
        found_models = {}
        
        # Search for original models
        if models_dir.exists():
            for key in ['cnn_original', 'dino_original']:
                matches = list(models_dir.glob(model_patterns[key]))
                if matches:
                    # Use the best one (highest accuracy if multiple)
                    best_model = max(matches, key=lambda x: self._extract_accuracy(x.name))
                    found_models[key] = str(best_model)
                    print(f"✅ Found {key}: {best_model.name}")
        
        # Search for quantized models
        if quantized_dir.exists():
            for key in ['cnn_quantized', 'dino_quantized']:
                matches = list(quantized_dir.glob(model_patterns[key]))
                if matches:
                    # Use the most recent one
                    latest_model = max(matches, key=lambda x: x.stat().st_mtime)
                    found_models[key] = str(latest_model)
                    print(f"✅ Found {key}: {latest_model.name}")
        
        return found_models
    
    def _extract_accuracy(self, filename):
        """Extract accuracy from filename for selecting best model."""
        import re
        match = re.search(r'acc(\d+\.?\d*)', filename)
        return float(match.group(1)) if match else 0.0
    
    def _load_model(self, model_path, force_cpu=False):
        """Load a single model (original or quantized)."""
        print(f"🔄 Loading model: {Path(model_path).name}")
        
        # Determine device
        eval_device = 'cpu' if force_cpu else self.device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=eval_device)
        
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        model_name = checkpoint.get('model_name', None)
        
        if model_name is None:
            # Try to infer from path
            if 'cnn' in str(model_path).lower() or 'efficientnet' in str(model_path).lower():
                model_name = 'efficientnet_b4'
            elif 'dino' in str(model_path).lower():
                model_name = 'dinov2_vits14'
            else:
                raise ValueError(f"Could not determine model type for {model_path}")
        
        # Get appropriate config
        if model_name.startswith('dinov2'):
            from src.config import config_dinov2
            config_module = config_dinov2
        elif model_name.startswith('efficientnet'):
            from src.config import config as config_cnn
            config_module = config_cnn
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Check if this is a quantized model
        is_quantized = 'quant_' in str(model_path) or 'quantized' in str(model_path)
        
        if is_quantized:
            # Load quantized model from saved data
            if 'model_state_dict' in checkpoint:
                # Create original model first
                original_model = create_model(num_classes, model_name, config_module)
                
                # Try to load quantized state
                try:
                    original_model.load_state_dict(checkpoint['model_state_dict'])
                    model = original_model
                    print(f"✅ Loaded quantized model state dict")
                except Exception as e:
                    print(f"⚠️ Could not load quantized state dict: {e}")
                    print("   Creating fresh quantized model...")
                    
                    # Create and quantize fresh model
                    model = create_model(num_classes, model_name, config_module)
                    model.eval()
                    model.cpu()
                    
                    # Apply dynamic quantization
                    if 'dino' in model_name.lower():
                        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                    else:
                        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
            else:
                # Legacy loading
                model = create_model(num_classes, model_name, config_module)
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Regular model loading
            model = create_model(num_classes, model_name, config_module)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(eval_device)
        
        model.eval()
        
        # Setup transforms
        transform = transforms.Compose([
            transforms.Resize(config_module.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        model_info = {
            'model_name': model_name,
            'class_names': class_names,
            'val_acc': checkpoint.get('val_acc', 0),
            'is_quantized': is_quantized,
            'is_transformer': 'dinov2' in model_name.lower(),
            'transform': transform,
            'config_module': config_module,
            'eval_device': eval_device
        }
        
        return model, model_info
    
    def load_all_models(self):
        """Load all available models for comparison."""
        found_models = self.find_models()
        
        print(f"\n🔍 Loading models for comparison...")
        
        for key, path in found_models.items():
            try:
                # Force CPU for quantized models
                force_cpu = 'quantized' in key
                model, info = self._load_model(path, force_cpu=force_cpu)
                self.models[key] = model
                self.model_info[key] = info
                print(f"✅ Loaded {key}: {info['model_name']} (quantized: {info['is_quantized']})")
            except Exception as e:
                print(f"❌ Failed to load {key}: {e}")
        
        print(f"\n📊 Successfully loaded {len(self.models)} models")
        return len(self.models) > 0
    
    def get_enhanced_transformer_attention(self, model, image_tensor, model_info):
        """Get more precise transformer attention with better focus."""
        B, C, H, W = image_tensor.shape
        patch_size = getattr(model.backbone, 'patch_size', 14) if hasattr(model, 'backbone') else 14
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Method 1: Multi-head attention rollout for better precision
        try:
            attention_weights = []
            
            def attention_hook(module, input, output):
                if len(input) > 0:
                    x = input[0]  # [batch, seq_len, dim]
                    B, N, C = x.shape
                    
                    if hasattr(module, 'qkv') and hasattr(module, 'num_heads') and hasattr(module, 'scale'):
                        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0], qkv[1], qkv[2]
                        
                        attn = (q @ k.transpose(-2, -1)) * module.scale
                        attn = attn.softmax(dim=-1)
                        attention_weights.append(attn.detach())
            
            # Get attention from last few blocks for better precision
            backbone = model.backbone if hasattr(model, 'backbone') else model
            if hasattr(backbone, 'blocks'):
                hooks = []
                # Use last 3 blocks for more robust attention
                for block in backbone.blocks[-3:]:
                    if hasattr(block, 'attn'):
                        hooks.append(block.attn.register_forward_hook(attention_hook))
            
            with torch.no_grad():
                _ = model(image_tensor)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            if attention_weights:
                # Average attention across blocks and heads
                all_attention = torch.stack(attention_weights)  # [blocks, batch, heads, seq, seq]
                avg_attention = all_attention.mean(0).mean(1)   # Average blocks and heads [batch, seq, seq]
                
                # Get CLS to patch attention
                cls_attn = avg_attention[0, 0, 1:]  # CLS token to patches
                
                if len(cls_attn) == num_patches:
                    attention_map = cls_attn.reshape(num_patches_h, num_patches_w)
                    
                    # Apply sharpening for more precise attention
                    attention_np = attention_map.cpu().numpy()
                    attention_np = np.power(attention_np, 1.5)  # Sharpen
                    
                    return attention_np
        except Exception as e:
            print(f"⚠️ Advanced transformer attention failed: {e}")
        
        # Fallback: Gradient-based attention
        return self.get_enhanced_gradient_attention(model, image_tensor, model_info)
    
    def get_enhanced_gradient_attention(self, model, image_tensor, model_info):
        """Enhanced gradient-based attention with better precision."""
        try:
            # Enable gradients
            image_tensor_grad = image_tensor.clone().detach().requires_grad_(True)
            
            # Forward pass
            output = model(image_tensor_grad)
            predicted_class = output.argmax(dim=1)
            class_score = output[0, predicted_class]
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=class_score,
                inputs=image_tensor_grad,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Enhanced gradient processing for better precision
            grad_magnitude = torch.sqrt(torch.sum(gradients[0]**2, dim=0))  # [H, W]
            
            # Apply guided backpropagation concept - keep only positive gradients
            positive_grads = torch.relu(gradients[0])  # [3, H, W]
            guided_magnitude = torch.sqrt(torch.sum(positive_grads**2, dim=0))
            
            # Combine regular and guided gradients
            combined = 0.7 * grad_magnitude + 0.3 * guided_magnitude
            
            # Apply sharpening and smoothing
            attention_np = combined.cpu().numpy()
            
            # Sharpen to improve precision
            attention_np = np.power(attention_np, 1.2)
            
            # Light smoothing to remove noise
            attention_np = cv2.GaussianBlur(attention_np, (3, 3), 0.5)
            
            # Normalize
            if attention_np.max() > attention_np.min():
                attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min())
            
            return attention_np
            
        except Exception as e:
            print(f"⚠️ Enhanced gradient attention failed: {e}")
            return None
    
    def create_precise_overlay(self, original_image, attention_map, alpha=0.5):
        """Create more precise attention overlay with better focus."""
        w, h = original_image.size
        
        # Resize with high-quality interpolation
        attention_resized = cv2.resize(attention_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply threshold to remove weak attention (improves precision)
        threshold = np.percentile(attention_resized, 75)  # Top 25%
        attention_focused = np.where(attention_resized > threshold, attention_resized, attention_resized * 0.3)
        
        # Light smoothing only
        attention_smooth = cv2.GaussianBlur(attention_focused, (5, 5), 0.8)
        
        # Normalize
        attention_norm = (attention_smooth - attention_smooth.min()) / (attention_smooth.max() - attention_smooth.min() + 1e-7)
        
        # Enhanced colormap for better precision visualization
        colors = ['#000022', '#000055', '#0000AA', '#0055FF', '#00AAFF', '#55FFAA', '#AAFF55', '#FFAA00', '#FF5500', '#FF0000']
        custom_cmap = LinearSegmentedColormap.from_list('precise_attention', colors, N=256)
        
        # Apply colormap
        heatmap = custom_cmap(attention_norm)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Convert original to numpy
        original_np = np.array(original_image)
        
        # Enhanced blending
        overlay = (1 - alpha) * original_np + alpha * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay, heatmap
    
    def analyze_image_with_model(self, image_path, model_key):
        """Analyze single image with specific model."""
        if model_key not in self.models:
            return None
        
        model = self.models[model_key]
        info = self.model_info[model_key]
        
        # Load and transform image
        original_image = Image.open(image_path).convert('RGB')
        image_tensor = info['transform'](original_image).unsqueeze(0)
        
        # Move to appropriate device
        if info['is_quantized']:
            image_tensor = image_tensor.cpu()
        else:
            image_tensor = image_tensor.to(info['eval_device'])
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
            predicted_class_idx = outputs.argmax(dim=1).item()
            confidence = probabilities[predicted_class_idx]
        
        # Generate enhanced attention
        if info['is_transformer']:
            attention_map = self.get_enhanced_transformer_attention(model, image_tensor, info)
        else:
            attention_map = self.get_enhanced_gradient_attention(model, image_tensor, info)
        
        if attention_map is not None:
            overlay, heatmap = self.create_precise_overlay(original_image, attention_map)
            has_attention = True
        else:
            overlay, heatmap = None, None
            has_attention = False
        
        return {
            'original_image': original_image,
            'attention_map': attention_map,
            'overlay': overlay,
            'heatmap': heatmap,
            'predicted_class': info['class_names'][predicted_class_idx],
            'confidence': confidence,
            'has_attention': has_attention,
            'model_name': info['model_name'],
            'is_quantized': info['is_quantized'],
            'is_transformer': info['is_transformer']
        }
    
    def create_comprehensive_comparison(self, image_paths, save_path=None):
        """Create comprehensive 4-model comparison visualization."""
        if not self.models:
            print("❌ No models loaded!")
            return
        
        n_images = len(image_paths)
        n_models = len(self.models)
        
        # Organize models for display: Original DINO, Original CNN, Quantized DINO, Quantized CNN
        model_order = ['dino_original', 'cnn_original', 'dino_quantized', 'cnn_quantized']
        available_models = [key for key in model_order if key in self.models]
        
        # Create figure
        fig, axes = plt.subplots(n_images, len(available_models) + 1, figsize=(4*(len(available_models)+1), 4*n_images))
        if n_images == 1:
            axes = axes.reshape(1, -1)
        if len(available_models) == 1:
            axes = axes.reshape(-1, 2)
        
        # Title
        title = "Enhanced Attention Comparison: Original vs Quantized Models"
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        for i, image_path in enumerate(image_paths):
            # Show original image first
            original_image = Image.open(image_path).convert('RGB')
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f'Original Image\n{Path(image_path).name}', fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Analyze with each model
            for j, model_key in enumerate(available_models):
                col_idx = j + 1
                result = self.analyze_image_with_model(image_path, model_key)
                
                if result:
                    # Create title with model info
                    model_type = "🤖 DINO" if result['is_transformer'] else "🖼️ CNN"
                    quant_status = "(Quantized)" if result['is_quantized'] else "(Original)"
                    title_text = f"{model_type} {quant_status}\nPred: {result['predicted_class']}\nConf: {result['confidence']:.3f}"
                    
                    # Show attention overlay or original
                    if result['has_attention'] and result['overlay'] is not None:
                        axes[i, col_idx].imshow(result['overlay'])
                        title_color = 'darkgreen' if result['confidence'] > 0.9 else 'darkblue'
                    else:
                        axes[i, col_idx].imshow(result['original_image'])
                        title_text += "\n⚠️ No Attention"
                        title_color = 'darkred'
                    
                    axes[i, col_idx].set_title(title_text, fontsize=9, fontweight='bold', color=title_color)
                    axes[i, col_idx].axis('off')
                    
                    # Print comparison info
                    device_info = "CPU" if result['is_quantized'] else "GPU/CPU"
                    print(f"🔍 {Path(image_path).name} - {model_key}: {result['predicted_class']} ({result['confidence']:.3f}) [{device_info}]")
                else:
                    axes[i, col_idx].text(0.5, 0.5, f'Model {model_key}\nFailed to Load', 
                                        ha='center', va='center', transform=axes[i, col_idx].transAxes)
                    axes[i, col_idx].set_title(f'{model_key}\n❌ Error', fontsize=10, color='red')
                    axes[i, col_idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"💾 Comprehensive comparison saved: {save_path}")
        
        plt.show()
        
        # Print summary
        print(f"\n📊 COMPARISON SUMMARY")
        print(f"{'='*50}")
        for key in available_models:
            info = self.model_info[key]
            model_type = "Transformer" if info['is_transformer'] else "CNN"
            status = "Quantized" if info['is_quantized'] else "Original"
            print(f"  {key}: {model_type} ({status}) - Acc: {info['val_acc']:.1%}")

def get_diverse_test_images(base_dir='crop_pest_data', n_images=3):
    """Get diverse test images representing different disease types."""
    import random
    
    image_dir = Path(base_dir)
    if not image_dir.exists():
        print(f"⚠️ Directory not found: {base_dir}")
        return []
    
    # Try to get images from different disease categories
    category_images = {}
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in image_dir.rglob(ext):
            category = img_path.parent.name
            if category not in category_images:
                category_images[category] = []
            category_images[category].append(str(img_path))
    
    # Sample from different categories
    selected_images = []
    random.seed(42)
    
    for category, images in list(category_images.items())[:n_images]:
        if images:
            selected_images.append(random.choice(images))
    
    # Fill remaining slots if needed
    while len(selected_images) < n_images:
        all_images = [img for imgs in category_images.values() for img in imgs]
        if all_images:
            remaining = [img for img in all_images if img not in selected_images]
            if remaining:
                selected_images.append(random.choice(remaining))
            else:
                break
        else:
            break
    
    return selected_images

def main():
    """Enhanced attention visualization with comprehensive model comparison."""
    print("🔬 ENHANCED ATTENTION VISUALIZATION WITH MODEL COMPARISON")
    print("="*70)
    
    # Initialize visualizer
    visualizer = EnhancedAttentionVisualizer()
    
    # Load all available models
    if not visualizer.load_all_models():
        print("❌ No models could be loaded!")
        return
    
    # Get diverse test images
    image_paths = get_diverse_test_images(n_images=3)
    if not image_paths:
        print("❌ No test images found!")
        return
    
    print(f"\n🖼️ Using test images:")
    for i, path in enumerate(image_paths, 1):
        print(f"  {i}. {Path(path).name} ({Path(path).parent.name})")
    
    # Create comprehensive comparison
    save_dir = Path(ATTENTION_VISUALIZATIONS_DIR)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = str(save_dir / "enhanced_attention_comparison.png")
    
    print(f"\n🎯 Creating comprehensive attention comparison...")
    visualizer.create_comprehensive_comparison(image_paths, save_path)
    
    print(f"\n✅ Enhanced attention visualization complete!")
    print(f"📁 Results saved to: {save_dir}")

if __name__ == "__main__":
    main() 