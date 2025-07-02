#!/usr/bin/env python3
"""
Simplified attention visualization using actual transformer attention weights.
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
sys.path.append('dino_cnn/thesis_codes')

from thesis_codes import model_setup
from thesis_codes import config as config_cnn
from thesis_codes import config_dinov2

class SimpleAttentionVisualizer:
    """Simple attention visualization using attention rollout for transformers."""
    
    def __init__(self, model_path, config_module):
        self.model_path = model_path
        self.config_module = config_module
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model, self.class_names, self.model_info = self._load_model()
        self.transform = self._setup_transforms()
        
        # Check if model is transformer-based
        self.is_transformer = 'dinov2' in self.model_info['model_name'].lower()
        
        print(f"✅ Model loaded: {self.model_info['model_name']}")
        print(f"📊 Classes: {len(self.class_names)}")
        print(f"🤖 Model type: {'Transformer' if self.is_transformer else 'CNN'}")
    
    def _load_model(self):
        """Load trained model."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        model_name = checkpoint.get('model_name', self.config_module.MODEL_NAME)
        
        model = model_setup.create_model(num_classes, model_name, self.config_module)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        model_info = {
            'model_name': model_name,
            'val_acc': checkpoint.get('val_acc', 0),
        }
        
        return model, class_names, model_info
    
    def _setup_transforms(self):
        """Setup image transforms."""
        return transforms.Compose([
            transforms.Resize(self.config_module.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_transformer_attention(self, image_tensor):
        """Extract attention weights from transformer model."""
        attentions = []
        
        def hook_fn(module, input, output):
            # For DINOv2 attention modules
            if hasattr(module, 'get_attention_map'):
                attentions.append(module.get_attention_map())
            # Try to capture attention from common patterns
            elif len(output) > 1 and hasattr(output[1], 'shape'):
                attentions.append(output[1])
        
        # Register hooks on attention modules
        hooks = []
        for name, module in self.model.named_modules():
            if 'attn' in name and 'proj' not in name:
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(image_tensor)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # If we got attention weights, process them
        if attentions:
            # Use the last attention layer
            attention = attentions[-1]
            if len(attention.shape) == 4:  # [batch, heads, seq, seq]
                # Average over heads and take [CLS] attention to patches
                attention = attention[0].mean(0)  # [seq, seq]
                cls_attention = attention[0, 1:]  # CLS to patches
                
                # Reshape to spatial grid
                num_patches = len(cls_attention)
                grid_size = int(np.sqrt(num_patches))
                if grid_size * grid_size == num_patches:
                    attention_map = cls_attention.reshape(grid_size, grid_size)
                    return attention_map.cpu().numpy()
        
        return None
    
    def get_cnn_attention(self, image_tensor, predicted_class):
        """Simple CNN attention using gradient magnitude."""
        # Enable gradients for input
        image_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, predicted_class]
        class_score.backward()
        
        # Get input gradients
        gradients = image_tensor.grad[0].cpu().numpy()  # [3, H, W]
        
        # Compute gradient magnitude across channels
        grad_magnitude = np.sqrt(np.sum(gradients**2, axis=0))
        
        # Normalize
        grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min() + 1e-7)
        
        return grad_magnitude
    
    def create_attention_overlay(self, original_image, attention_map, alpha=0.4):
        """Create smooth attention overlay."""
        # Get original image size
        w, h = original_image.size
        
        # Resize attention map with smooth interpolation
        attention_resized = cv2.resize(attention_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply strong Gaussian smoothing for better visualization
        attention_smooth = cv2.GaussianBlur(attention_resized, (15, 15), 0)
        
        # Normalize
        attention_norm = (attention_smooth - attention_smooth.min()) / (attention_smooth.max() - attention_smooth.min() + 1e-7)
        
        # Create a nicer colormap
        colors = ['#000044', '#000088', '#0000FF', '#0088FF', '#00FFFF', '#88FF88', '#FFFF00', '#FF8800', '#FF0000']
        custom_cmap = LinearSegmentedColormap.from_list('attention', colors, N=256)
        
        # Apply colormap
        heatmap = custom_cmap(attention_norm)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Convert original to numpy
        original_np = np.array(original_image)
        
        # Blend with lower alpha for more subtle effect
        overlay = (1 - alpha) * original_np + alpha * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay, heatmap
    
    def analyze_image(self, image_path):
        """Analyze image and generate attention."""
        # Load image
        original_image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
            predicted_class_idx = outputs.argmax(dim=1).item()
            confidence = probabilities[predicted_class_idx]
        
        # Generate attention based on model type
        if self.is_transformer:
            print("🔍 Extracting transformer attention...")
            attention_map = self.get_transformer_attention(image_tensor)
        else:
            print("🔍 Computing CNN gradient attention...")
            attention_map = self.get_cnn_attention(image_tensor, predicted_class_idx)
        
        if attention_map is not None:
            overlay, heatmap = self.create_attention_overlay(original_image, attention_map)
            has_attention = True
        else:
            print("⚠️ Could not generate attention map")
            overlay, heatmap = None, None
            has_attention = False
        
        return {
            'original_image': original_image,
            'attention_map': attention_map,
            'overlay': overlay,
            'heatmap': heatmap,
            'predicted_class': self.class_names[predicted_class_idx],
            'confidence': confidence,
            'has_attention': has_attention
        }
    
    def visualize_images(self, image_paths, save_path=None):
        """Visualize attention for multiple images."""
        n_images = len(image_paths)
        fig, axes = plt.subplots(3, n_images, figsize=(4*n_images, 10))
        if n_images == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'{self.model_info["model_name"]} - Simple Attention Visualization', 
                    fontsize=16, fontweight='bold')
        
        for i, image_path in enumerate(image_paths):
            result = self.analyze_image(image_path)
            
            # Original image
            axes[0, i].imshow(result['original_image'])
            axes[0, i].set_title(f'Original\n{Path(image_path).name}', fontsize=10)
            axes[0, i].axis('off')
            
            # Attention overlay
            if result['has_attention'] and result['overlay'] is not None:
                axes[1, i].imshow(result['overlay'])
                axes[1, i].set_title(f'Attention Overlay\nPred: {result["predicted_class"]}\nConf: {result["confidence"]:.3f}', 
                                   fontsize=10, fontweight='bold')
            else:
                axes[1, i].imshow(result['original_image'])
                axes[1, i].set_title(f'No Attention\nPred: {result["predicted_class"]}\nConf: {result["confidence"]:.3f}', 
                                   fontsize=10)
            axes[1, i].axis('off')
            
            # Pure heatmap
            if result['has_attention'] and result['heatmap'] is not None:
                axes[2, i].imshow(result['heatmap'])
                axes[2, i].set_title('Attention Heatmap', fontsize=10)
            else:
                axes[2, i].text(0.5, 0.5, 'No Heatmap\nAvailable', 
                              ha='center', va='center', transform=axes[2, i].transAxes, fontsize=12)
                axes[2, i].set_title('No Heatmap Available', fontsize=10)
            axes[2, i].axis('off')
            
            print(f"🔍 {Path(image_path).name}: {result['predicted_class']} ({result['confidence']:.3f})")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")
        
        plt.show()

def get_random_images(base_dir='crop_pest_data', n_images=3):
    """Get random test images."""
    import random
    
    image_dir = Path(base_dir)
    if not image_dir.exists():
        print(f"⚠️ Directory not found: {base_dir}")
        return []
    
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(image_dir.rglob(ext))
    
    if len(image_files) < n_images:
        print(f"⚠️ Only found {len(image_files)} images, using all")
        return [str(f) for f in image_files]
    
    # Sample random images
    random.seed(42)
    selected = random.sample(image_files, n_images)
    return [str(f) for f in selected]

def main():
    """Test both models with simple attention."""
    models = [
        ('models/best_cnn_b4_acc73.40_20250702_133941.pth', config_cnn),
        ('models/best_dino_vits14_acc74.47_20250702_134026.pth', config_dinov2)
    ]
    
    # Get test images
    image_paths = get_random_images(n_images=3)
    if not image_paths:
        print("❌ No test images found!")
        return
    
    print(f"🖼️ Using images: {[Path(p).name for p in image_paths]}")
    
    for model_path, config_module in models:
        if not Path(model_path).exists():
            print(f"⚠️ Model not found: {model_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing {Path(model_path).name}")
        print(f"{'='*60}")
        
        try:
            visualizer = SimpleAttentionVisualizer(model_path, config_module)
            
            # Create save path
            model_name = visualizer.model_info['model_name']
            save_path = f"attention_visualizations/simple_{model_name}_attention.png"
            Path("attention_visualizations").mkdir(exist_ok=True)
            
            # Visualize
            visualizer.visualize_images(image_paths, save_path)
            
        except Exception as e:
            print(f"❌ Error with {model_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 