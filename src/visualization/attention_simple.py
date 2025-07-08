#!/usr/bin/env python3
"""
Enhanced attention visualization for transformer and CNN models.
Focus on interpretable attention maps with multiple techniques:
- Multi-head averaging
- Mid-layer attention
- Token attention visualization  
- Smoothed attention maps
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

class ImprovedAttentionVisualizer:
    """Improved attention visualization with better interpretability techniques."""
    
    def __init__(self, temperature=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}  # Will store loaded models
        self.model_info = {}
        self.temperature = temperature  # For softmax smoothing
        print(f"🚀 Using device: {self.device}")
        
        # Disable xFormers for CPU inference to avoid compatibility issues
        if self.device.type == 'cpu':
            import os
            os.environ['XFORMERS_DISABLED'] = '1'
            print("⚠️ Disabled xFormers for CPU inference")
        
    def find_models(self):
        """Auto-detect available original models only."""
        models_dir = Path(PROJECT_ROOT) / 'models'
        
        model_patterns = {
            'cnn_original': 'best_cnn_*_acc*.pth',
            'dino_original': 'best_dino_*_acc*.pth',
        }
        
        found_models = {}
        
        # Search for original models only
        if models_dir.exists():
            for key in ['cnn_original', 'dino_original']:
                matches = list(models_dir.glob(model_patterns[key]))
                if matches:
                    # Use the best one (highest accuracy if multiple)
                    best_model = max(matches, key=lambda x: self._extract_accuracy(x.name))
                    found_models[key] = str(best_model)
                    print(f"✅ Found {key}: {best_model.name}")
        
        return found_models
    
    def _extract_accuracy(self, filename):
        """Extract accuracy from filename for selecting best model."""
        import re
        match = re.search(r'acc(\d+\.?\d*)', filename)
        return float(match.group(1)) if match else 0.0
    
    def _load_model(self, model_path):
        """Load a single original model."""
        print(f"🔄 Loading model: {Path(model_path).name}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
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
        
        # Load model
        model = create_model(num_classes, model_name, config_module)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # For DINOv2 models on CPU, we need to handle xFormers compatibility
        if 'dinov2' in model_name.lower() and self.device.type == 'cpu':
            print("⚠️ DINOv2 on CPU - ensuring compatibility...")
            # Force the model to use regular attention instead of xFormers
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
                for block in model.backbone.blocks:
                    if hasattr(block.attn, 'fused_attn'):
                        block.attn.fused_attn = False
        
        model = model.to(self.device)
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
            'is_transformer': 'dinov2' in model_name.lower(),
            'transform': transform,
            'config_module': config_module,
        }
        
        return model, model_info
    
    def load_all_models(self):
        """Load all available original models."""
        found_models = self.find_models()
        
        print(f"\n🔍 Loading models for visualization...")
        
        for key, path in found_models.items():
            try:
                model, info = self._load_model(path)
                self.models[key] = model
                self.model_info[key] = info
                print(f"✅ Loaded {key}: {info['model_name']}")
            except Exception as e:
                print(f"❌ Failed to load {key}: {e}")
        
        print(f"\n📊 Successfully loaded {len(self.models)} models")
        return len(self.models) > 0
    
    def get_improved_transformer_attention(self, model, image_tensor, model_info, target_layer=6):
        """
        Get improved transformer attention using LLM suggestions:
        1. Average attention across all heads
        2. Use mid-layer attention (layer 4-6) instead of last layer
        3. Visualize token attention, not just class token
        4. Use softmax with temperature for smoothing
        """
        B, C, H, W = image_tensor.shape
        patch_size = getattr(model.backbone, 'patch_size', 14) if hasattr(model, 'backbone') else 14
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w
        
        try:
            attention_weights = []
            
            def attention_hook(module, input, output):
                if len(input) > 0:
                    x = input[0]  # [batch, seq_len, dim]
                    B, N, C = x.shape
                    
                    if hasattr(module, 'qkv') and hasattr(module, 'num_heads') and hasattr(module, 'scale'):
                        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0], qkv[1], qkv[2]
                        
                        # Apply temperature for smoothing before softmax
                        attn = (q @ k.transpose(-2, -1)) * module.scale / self.temperature
                        attn = attn.softmax(dim=-1)
                        attention_weights.append(attn.detach())
            
            # Use mid-layer attention for better interpretability
            backbone = model.backbone if hasattr(model, 'backbone') else model
            if hasattr(backbone, 'blocks'):
                # Use specific layer (default layer 6 out of 12)
                target_block = min(target_layer, len(backbone.blocks) - 1)
                hook = backbone.blocks[target_block].attn.register_forward_hook(attention_hook)
            
            # Ensure model and input are on the same device
            current_device = next(model.parameters()).device
            image_tensor = image_tensor.to(current_device)
            
            with torch.no_grad():
                try:
                    _ = model(image_tensor)
                except Exception as e:
                    if "memory_efficient_attention" in str(e) or "xformers" in str(e).lower():
                        print(f"⚠️ xFormers issue in attention hook, skipping transformer attention")
                        return None
                    else:
                        raise e
            
            # Remove hook
            hook.remove()
            
            if attention_weights:
                # Average attention across all heads (suggestion 1)
                attention = attention_weights[0]  # [batch, heads, seq, seq]
                avg_attention = attention.mean(dim=1)  # [batch, seq, seq]
                
                # Method 1: Class token attention to patches
                cls_attn = avg_attention[0, 0, 1:]  # CLS token to patches
                
                # Method 2: Average token attention (suggestion 3)
                # Get attention from all patch tokens to all patches
                patch_to_patch = avg_attention[0, 1:, 1:]  # [patches, patches]
                token_attn = patch_to_patch.mean(dim=0)  # Average over source tokens
                
                # Combine both methods for more robust visualization
                combined_attn = 0.6 * cls_attn + 0.4 * token_attn
                
                if len(combined_attn) == num_patches:
                    attention_map = combined_attn.reshape(num_patches_h, num_patches_w)
                    attention_np = attention_map.cpu().numpy()
                    
                    # Apply softmax-sum for smoothing (suggestion 5)
                    attention_smoothed = F.softmax(torch.from_numpy(attention_np) / self.temperature, dim=-1)
                    attention_np = attention_smoothed.numpy()
                    
                    return attention_np
        except Exception as e:
            print(f"⚠️ Improved transformer attention failed: {e}")
        
        # Fallback: Gradient-based attention
        return self.get_improved_gradient_attention(model, image_tensor, model_info)
    
    def get_improved_gradient_attention(self, model, image_tensor, model_info):
        """Enhanced gradient-based attention with better precision."""
        try:
            # Ensure model and input are on the same device
            current_device = next(model.parameters()).device
            image_tensor = image_tensor.to(current_device)
            
            # Enable gradients
            image_tensor_grad = image_tensor.clone().detach().requires_grad_(True)
            
            # Forward pass with error handling
            try:
                output = model(image_tensor_grad)
            except Exception as e:
                if "memory_efficient_attention" in str(e) or "xformers" in str(e).lower():
                    print(f"⚠️ xFormers issue in gradient attention, moving to CPU")
                    model = model.cpu()
                    image_tensor_grad = image_tensor_grad.cpu()
                    output = model(image_tensor_grad)
                else:
                    raise e
            predicted_class = output.argmax(dim=1)
            class_score = output[0, predicted_class]
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=class_score,
                inputs=image_tensor_grad,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Enhanced gradient processing
            grad_magnitude = torch.sqrt(torch.sum(gradients[0]**2, dim=0))  # [H, W]
            
            # Apply guided backpropagation - keep only positive gradients
            positive_grads = torch.relu(gradients[0])  # [3, H, W]
            guided_magnitude = torch.sqrt(torch.sum(positive_grads**2, dim=0))
            
            # Combine regular and guided gradients
            combined = 0.7 * grad_magnitude + 0.3 * guided_magnitude
            
            # Apply smoothing with temperature
            attention_tensor = combined / self.temperature
            attention_tensor = F.softmax(attention_tensor.flatten(), dim=0).reshape(combined.shape)
            
            attention_np = attention_tensor.cpu().numpy()
            
            # Light gaussian smoothing
            attention_np = cv2.GaussianBlur(attention_np, (3, 3), 0.5)
            
            # Normalize
            if attention_np.max() > attention_np.min():
                attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min())
            
            return attention_np
            
        except Exception as e:
            print(f"⚠️ Improved gradient attention failed: {e}")
            return None
    
    def create_improved_overlay(self, original_image, attention_map, alpha=0.6):
        """Create improved attention overlay with better visualization."""
        w, h = original_image.size
        
        # Resize with high-quality interpolation
        attention_resized = cv2.resize(attention_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply percentile thresholding for better focus
        threshold = np.percentile(attention_resized, 60)  # Top 40%
        attention_focused = np.where(attention_resized > threshold, attention_resized, attention_resized * 0.2)
        
        # Light smoothing
        attention_smooth = cv2.GaussianBlur(attention_focused, (5, 5), 1.0)
        
        # Normalize
        attention_norm = (attention_smooth - attention_smooth.min()) / (attention_smooth.max() - attention_smooth.min() + 1e-7)
        
        # Better colormap for attention visualization
        colors = ['#000033', '#000066', '#003399', '#0066CC', '#0099FF', '#33CCFF', '#66FFCC', '#99FF99', '#CCFF66', '#FFCC00', '#FF9900', '#FF6600', '#FF3300']
        custom_cmap = LinearSegmentedColormap.from_list('improved_attention', colors, N=256)
        
        # Apply colormap
        heatmap = custom_cmap(attention_norm)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Convert original to numpy
        original_np = np.array(original_image)
        
        # Improved blending with better alpha
        overlay = (1 - alpha) * original_np + alpha * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay, heatmap
    
    def analyze_image_with_model(self, image_path, model_key, overlay_mode='overlay'):
        """Analyze single image with specific model using improved methods."""
        if model_key not in self.models:
            return None
        
        model = self.models[model_key]
        info = self.model_info[model_key]
        
        # Load and transform image
        original_image = Image.open(image_path).convert('RGB')
        image_tensor = info['transform'](original_image).unsqueeze(0).to(self.device)
        
        # Get prediction with error handling for xFormers issues
        try:
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
                predicted_class_idx = outputs.argmax(dim=1).item()
                confidence = probabilities[predicted_class_idx]
        except Exception as e:
            if "memory_efficient_attention" in str(e) or "xformers" in str(e).lower():
                print(f"⚠️ xFormers issue detected, trying CPU fallback for {model_key}")
                # Move everything to CPU and try again
                model = model.cpu()
                image_tensor = image_tensor.cpu()
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
                    predicted_class_idx = outputs.argmax(dim=1).item()
                    confidence = probabilities[predicted_class_idx]
            else:
                raise e
        
        # Generate attention
        if info['is_transformer']:
            attention_map = self.get_improved_transformer_attention(model, image_tensor, info)
        else:
            # Prefer Grad-CAM for CNNs
            attention_map = self.get_gradcam_attention(model, image_tensor)
            if attention_map is None:
                # Fallback to gradient magnitude if Grad-CAM fails
                attention_map = self.get_improved_gradient_attention(model, image_tensor, info)
        
        if attention_map is not None:
            if overlay_mode == 'overlay':
                overlay, heatmap = self.create_improved_overlay(original_image, attention_map)
            elif overlay_mode == 'dot':
                overlay, heatmap = self.create_dot_overlay(original_image, attention_map)
            else: # Default to overlay
                overlay, heatmap = self.create_improved_overlay(original_image, attention_map)
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
            'is_transformer': info['is_transformer']
        }
    
    def create_simple_visualization(self, image_paths, save_path=None, overlay_mode='overlay'):
        """Create simple but effective attention visualization without quantized comparison."""
        if not self.models:
            print("❌ No models loaded!")
            return
        
        n_images = len(image_paths)
        available_models = list(self.models.keys())
        
        # Create figure: Original | Model 1 (Overlay + Heatmap) | Model 2 (Overlay + Heatmap)
        n_cols = 1 + len(available_models) * 2  # Original + (Overlay + Heatmap) per model
        fig, axes = plt.subplots(n_images, n_cols, figsize=(4*n_cols, 4*n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Title
        title = f"Enhanced Attention Visualization - Simple Method"
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        for i, image_path in enumerate(image_paths):
            # Show original image first
            original_image = Image.open(image_path).convert('RGB')
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f'Original\n{Path(image_path).name}', fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Analyze with each model
            col_idx = 1
            for model_key in available_models:
                result = self.analyze_image_with_model(image_path, model_key, overlay_mode)
                
                if result:
                    # Model info
                    model_type = "🤖 DINOv2" if result['is_transformer'] else "🖼️ EfficientNet"
                    
                    # Show attention overlay
                    if result['has_attention'] and result['overlay'] is not None:
                        axes[i, col_idx].imshow(result['overlay'])
                        title_text = f"{model_type}\nPred: {result['predicted_class']}\nConf: {result['confidence']:.3f}"
                        title_color = 'darkgreen' if result['confidence'] > 0.9 else 'darkblue'
                    else:
                        axes[i, col_idx].imshow(result['original_image'])
                        title_text = f"{model_type}\n⚠️ No Attention"
                        title_color = 'darkred'
                    
                    axes[i, col_idx].set_title(f'Attention Overlay\n{title_text}', fontsize=9, fontweight='bold', color=title_color)
                    axes[i, col_idx].axis('off')
                    
                    # Show attention heatmap
                    if result['has_attention'] and result['heatmap'] is not None:
                        axes[i, col_idx + 1].imshow(result['heatmap'])
                        axes[i, col_idx + 1].set_title('Attention Heatmap', fontsize=9, fontweight='bold')
                    else:
                        axes[i, col_idx + 1].text(0.5, 0.5, 'No Heatmap', ha='center', va='center', 
                                                transform=axes[i, col_idx + 1].transAxes)
                        axes[i, col_idx + 1].set_title('No Heatmap', fontsize=9, color='red')
                    
                    axes[i, col_idx + 1].axis('off')
                    
                    # Print analysis info
                    print(f"🔍 {Path(image_path).name} - {model_key}: {result['predicted_class']} ({result['confidence']:.3f})")
                
                else:
                    # Error handling
                    for j in range(2):  # Overlay and heatmap columns
                        axes[i, col_idx + j].text(0.5, 0.5, f'Model {model_key}\nFailed to Load', 
                                                ha='center', va='center', transform=axes[i, col_idx + j].transAxes)
                        axes[i, col_idx + j].set_title(f'{model_key}\n❌ Error', fontsize=10, color='red')
                        axes[i, col_idx + j].axis('off')
                
                col_idx += 2  # Move to next model (overlay + heatmap)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"💾 Simple attention visualization saved: {save_path}")

        # ------------------------------
        # Export attention maps for further analysis
        # ------------------------------
        export_dir = Path(ATTENTION_VISUALIZATIONS_DIR) / "attention_maps"
        for i, image_path in enumerate(image_paths):
            for model_key in available_models:
                result = self.analyze_image_with_model(image_path, model_key, overlay_mode)
                if result and result['attention_map'] is not None:
                    self.save_attention_maps(image_path, model_key, result['attention_map'], result.get('heatmap'))
         
        plt.show()

        # After combined visualization, create individual model variants
        for model_key in available_models:
            variant_path = None
            if save_path:
                base = Path(save_path)
                variant_path = str(base.with_name(f"{model_key}_visualization.png"))
            self.create_model_specific_visualization(image_paths, model_key, variant_path, overlay_mode)
        
        # Print summary
        print(f"\n📊 ATTENTION VISUALIZATION SUMMARY")
        print(f"{'='*50}")
        for key in available_models:
            info = self.model_info[key]
            model_type = "Transformer (DINOv2)" if info['is_transformer'] else "CNN (EfficientNet)"
            val_acc = info['val_acc']
            # Fix formatting: if >1 assume already percentage
            if val_acc > 1:
                acc_display = f"{val_acc:.2f}%"
            else:
                acc_display = f"{val_acc*100:.2f}%"
            print(f"  {key}: {model_type} - Acc: {acc_display}")

    def save_attention_maps(self, image_path, model_key, attention_map, heatmap=None):
        """Save raw attention maps (npy) and heatmap PNG for downstream analysis."""
        out_dir = Path(ATTENTION_VISUALIZATIONS_DIR) / "attention_maps" / model_key
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(image_path).stem
        # Save raw attention tensor
        np.save(out_dir / f"{base_name}_attn.npy", attention_map)
        # Save heatmap as PNG for quick viewing
        if heatmap is not None:
            heatmap_img = Image.fromarray(heatmap)
            heatmap_img.save(out_dir / f"{base_name}_heatmap.png")

    def create_model_specific_visualization(self, image_paths, model_key, save_path=None, overlay_mode='overlay'):
        """Generate visualization for one model only (Original, Overlay, Heatmap)."""
        if model_key not in self.models:
            print(f"⚠️ Model {model_key} not loaded; skipping model-specific visualization.")
            return
        n_images = len(image_paths)
        fig, axes = plt.subplots(n_images, 3, figsize=(12, 4*n_images))
        if n_images == 1:
            axes = axes.reshape(1, -1)
        model_type = "DINOv2" if self.model_info[model_key]['is_transformer'] else "EfficientNet"
        fig.suptitle(f"{model_type} - Simple Attention Visualization", fontsize=16, fontweight='bold', y=0.98)
        for i, img_path in enumerate(image_paths):
            original_img = Image.open(img_path).convert('RGB')
            axes[i,0].imshow(original_img)
            axes[i,0].set_title('Original', fontsize=10, fontweight='bold')
            axes[i,0].axis('off')
            # Analyze with model (caches already loaded models)
            result = self.analyze_image_with_model(img_path, model_key, overlay_mode)
            if result and result['overlay'] is not None:
                axes[i,1].imshow(result['overlay'])
                axes[i,1].set_title(f"Attention Overlay\nPred: {result['predicted_class']}\nConf: {result['confidence']:.3f}", fontsize=9, fontweight='bold')
            else:
                axes[i,1].imshow(original_img)
                axes[i,1].set_title('No Attention', fontsize=9, color='red')
            axes[i,1].axis('off')
            if result and result['heatmap'] is not None:
                axes[i,2].imshow(result['heatmap'])
                axes[i,2].set_title('Attention Heatmap', fontsize=9, fontweight='bold')
            else:
                axes[i,2].text(0.5,0.5,'No Heatmap', ha='center', va='center', transform=axes[i,2].transAxes, color='red')
                axes[i,2].set_title('No Heatmap', fontsize=9, color='red')
            axes[i,2].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"💾 Model-specific visualization saved: {save_path}")
        plt.show()

    def _find_last_conv(self, model):
        """Utility to find the last Conv2d layer in a model (for Grad-CAM)."""
        # Accept any module whose class name contains "Conv" and has weight with 4 dims
        for module in reversed(list(model.modules())):
            if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                if module.weight.ndim == 4:
                    return module
        return None

    def get_gradcam_attention(self, model, image_tensor):
        """Compute Grad-CAM for CNN models to obtain spatial attention."""
        target_layer = self._find_last_conv(model)
        if target_layer is None:
            print("⚠️ No Conv2d layer found for Grad-CAM; falling back to gradient magnitude.")
            return None
        activations = {}
        gradients = {}
        
        def forward_hook(module, inp, out):
            activations['value'] = out.detach()
        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple with same shape as output
            gradients['value'] = grad_out[0].detach()

        fh = target_layer.register_forward_hook(forward_hook)
        # Use full backward hook if available (PyTorch >=1.12)
        if hasattr(target_layer, 'register_full_backward_hook'):
            bh = target_layer.register_full_backward_hook(backward_hook)
        else:
            bh = target_layer.register_backward_hook(backward_hook)
        
        # Forward
        # Enable grad on input for safety
        image_tensor = image_tensor.requires_grad_(True)
        outputs = model(image_tensor)
        pred_idx = outputs.argmax(dim=1)
        score = outputs[0, pred_idx]
        model.zero_grad()
        score.backward()
        
        fh.remove(); bh.remove()
        if 'value' not in activations or 'value' not in gradients:
            print("⚠️ Grad-CAM hooks failed; returning None")
            return None
        act = activations['value']  # [B, C, H, W]
        grad = gradients['value']   # same shape
        weights = grad.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
        cam = (weights * act).sum(dim=1, keepdim=False)  # [B,H,W]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam[0].cpu().numpy()
        return cam_np

    def create_dot_overlay(self, original_image, attention_map, num_dots=50, color=(255,0,0)):
        """Overlay top-K attention regions as colored dots for clarity."""
        w,h = original_image.size
        attn_resized = cv2.resize(attention_map.astype(np.float32), (w,h), interpolation=cv2.INTER_LANCZOS4)
        flat = attn_resized.flatten()
        k = max(1, int(len(flat) * 0.01))  # top 1% pixels by default
        topk_idx = np.argpartition(flat, -k)[-k:]
        ys, xs = np.unravel_index(topk_idx, attn_resized.shape)
        overlay = np.array(original_image).copy()
        for x,y in zip(xs,ys):
            cv2.circle(overlay, (x,y), 5, color, thickness=-1, lineType=cv2.LINE_AA)
        return overlay, None

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
    """Enhanced attention visualization with improved methods."""
    print("🔬 ENHANCED ATTENTION VISUALIZATION")
    print("="*50)
    
    # Initialize visualizer with temperature for smoothing
    visualizer = ImprovedAttentionVisualizer(temperature=0.7)
    
    # Load all available original models (no quantized)
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
    
    # Create simple but effective visualization
    save_dir = Path(ATTENTION_VISUALIZATIONS_DIR)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = str(save_dir / "simple_attention_visualization.png")
    
    print(f"\n🎯 Creating improved attention visualization...")
    visualizer.create_simple_visualization(image_paths, save_path)
    
    print(f"\n✅ Enhanced attention visualization complete!")
    print(f"📁 Results saved to: {save_dir}")

if __name__ == "__main__":
    main() 