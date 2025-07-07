#!/usr/bin/env python3
"""
🔍 Model Visualization Toolkit
============================

GradCAM, Feature Maps, and Attention Heat Maps for understanding model behavior.
Perfect for analyzing the tomato disease confusion issue!

Usage:
    python model_visualization.py --model path/to/model.pth --images path/to/test/images/
    
Key Features:
- GradCAM for any CNN layer
- Feature map visualization 
- Attention heat maps (for Vision Transformers)
- Side-by-side comparisons
- Batch processing
- Confusion analysis mode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import argparse
import json
from pathlib import Path
import cv2  # type: ignore
from torchvision import transforms
from typing import List, Dict, Tuple, Optional, cast
import warnings
warnings.filterwarnings('ignore')

# Import configurations  
from ..config import config_paths
from ..config import config
from ..config import config_dinov2
from ..models import model_setup
from ..data import dataloader_setup
from ..data import data_splitter


class GradCAM:
    """GradCAM implementation for CNN models."""
    
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks: List = []
        
        # Auto-detect target layer if not specified
        if target_layer_name is None:
            self.target_layer_name = self._find_best_layer()
        
        self._register_hooks()
    
    def _find_best_layer(self):
        """Auto-detect the best layer for GradCAM (usually last conv layer)."""
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                conv_layers.append(name)
        
        if conv_layers:
            best_layer = conv_layers[-1]  # Last conv layer
            print(f"🎯 Auto-detected GradCAM target layer: {best_layer}")
            return best_layer
        else:
            # Fallback for other architectures
            all_layers = [name for name, _ in self.model.named_modules()]
            if all_layers:
                best_layer = all_layers[-2]  # Second to last layer
                print(f"🎯 Using fallback layer: {best_layer}")
                return best_layer
            else:
                raise ValueError("Could not find suitable layer for GradCAM")
    
    def _get_module_by_name(self, name):
        """Get module by its name."""
        modules = dict(self.model.named_modules())
        return modules.get(name)
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        target_module = self._get_module_by_name(self.target_layer_name)
        if target_module is None:
            available_layers = [name for name, _ in self.model.named_modules()]
            raise ValueError(f"Layer '{self.target_layer_name}' not found. Available: {available_layers[:10]}...")
        
        # Forward hook to save activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Backward hook to save gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hooks.append(target_module.register_forward_hook(forward_hook))
        self.hooks.append(target_module.register_backward_hook(backward_hook))
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate CAM for given input and class."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Generate CAM
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured. Make sure hooks are registered correctly.")
        
        gradients = self.gradients[0]  # Shape: [C, H, W]
        activations = self.activations[0]  # Shape: [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=[1, 2])  # Shape: [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam.cpu().numpy(), output.softmax(dim=1)[0].cpu().numpy(), class_idx
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class FeatureMapVisualizer:
    """Visualize intermediate feature maps from CNN layers."""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self, layer_names: List[str]):
        """Register hooks for specified layers."""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        modules = dict(self.model.named_modules())
        for name in layer_names:
            if name in modules:
                hook = modules[name].register_forward_hook(get_activation(name))
                self.hooks.append(hook)
            else:
                print(f"⚠️  Layer '{name}' not found")
    
    def get_feature_maps(self, input_tensor, layer_name):
        """Get feature maps for a specific layer."""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if layer_name in self.activations:
            return self.activations[layer_name].cpu().numpy()
        else:
            return None
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class AttentionVisualizer:
    """Visualize attention maps for Vision Transformers (DINOv2)."""
    
    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self.hooks = []
    
    def register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        def get_attention(name):
            def hook(module, input, output):
                # For multi-head attention, we usually want the attention weights
                if hasattr(module, 'attn_drop'):  # Typical attention module
                    # Store attention weights if available
                    self.attention_maps[name] = output.detach()
            return hook
        
        # Look for attention modules
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                hook = module.register_forward_hook(get_attention(name))
                self.hooks.append(hook)
                print(f"🎯 Registered attention hook: {name}")
    
    def get_attention_maps(self, input_tensor):
        """Get attention maps."""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        return self.attention_maps
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class ModelVisualizationPipeline:
    """Main pipeline for model visualization and analysis."""
    
    def __init__(self, model_path: str, config_module=None):
        """Initialize the visualization pipeline."""
        print(f"🔍 Setting up Model Visualization Pipeline...")
        
        self.model_path = model_path
        self.config = config_module or config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and setup components
        self.model, self.class_names = self._load_model()
        self.transform = self._setup_transforms()
        
        # Initialize visualization tools
        self.gradcam = GradCAM(self.model)
        self.feature_viz = FeatureMapVisualizer(self.model)
        self.attention_viz = AttentionVisualizer(self.model)
        
        print(f"✅ Pipeline ready!")
        print(f"🎯 Model: {Path(model_path).name}")
        print(f"🏷️  Classes: {len(self.class_names)}")
        print(f"💻 Device: {self.device}")
    
    def _load_model(self):
        """Load trained model and class names."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model info from checkpoint
        num_classes = len(checkpoint['class_names'])
        # Determine default model based on config module name
        config_name = getattr(self.config, '__name__', '').split('.')[-1]
        default_model = 'dinov2_vits14' if 'dinov2' in config_name else 'efficientnet_b4'
        model_name = checkpoint.get('model_name', getattr(self.config, 'MODEL_NAME', default_model))
        
        # Create model
        model = model_setup.create_model(num_classes, model_name, self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model, checkpoint['class_names']
    
    def _setup_transforms(self):
        """Setup image transforms."""
        image_size = getattr(self.config, 'IMAGE_SIZE', (224, 224))
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def analyze_single_image(self, image_path: str, save_dir: Optional[str] = None, 
                           show_feature_maps: bool = True, show_gradcam: bool = True,
                           show_attention: bool = False) -> Dict:
        """
        Comprehensive analysis of a single image with multiple visualization techniques.
        
        Args:
            image_path: Path to the input image
            save_dir: Directory to save visualizations (defaults to centralized visualizations directory)
            show_feature_maps: Whether to generate feature map visualizations
            show_gradcam: Whether to generate GradCAM visualizations  
            show_attention: Whether to generate attention visualizations
            
        Returns:
            Dictionary containing analysis results and paths to saved visualizations
        """
        # Use centralized visualization directory if no save_dir specified
        if save_dir is None:
            config_paths.ensure_directories()
            save_dir = config_paths.VISUALIZATIONS_DIR
        else:
            os.makedirs(save_dir, exist_ok=True)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_preprocessed = cast(torch.Tensor, self.transform(image))
        image_tensor = image_preprocessed.unsqueeze(0).to(self.device)
        
        results = {
            'image_path': image_path,
            'original_image': image,
            'predictions': {},
            'visualizations': {}
        }
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = output.softmax(dim=1)[0].cpu().numpy()
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[predicted_class]
        
        results['predictions'] = {
            'predicted_class': self.class_names[predicted_class],
            'predicted_idx': predicted_class,
            'confidence': float(confidence),
            'top_5': [(self.class_names[i], float(probabilities[i])) 
                     for i in probabilities.argsort()[-5:][::-1]]
        }
        
        print(f"🔍 Analyzing: {Path(image_path).name}")
        print(f"🎯 Prediction: {self.class_names[predicted_class]} ({confidence:.3f})")
        
        # GradCAM
        if show_gradcam:
            cam, _, _ = self.gradcam.generate_cam(image_tensor, predicted_class)
            gradcam_viz = self._create_gradcam_visualization(image, cam)
            results['visualizations']['gradcam'] = gradcam_viz
            
            # Save GradCAM
            gradcam_path = os.path.join(save_dir, f"gradcam_{Path(image_path).stem}.png")
            gradcam_viz.save(gradcam_path)
            print(f"💾 GradCAM saved: {gradcam_path}")
        
        # Feature Maps
        if show_feature_maps:
            feature_maps = self._extract_key_feature_maps(image_tensor)
            if feature_maps:
                feature_viz_path = os.path.join(save_dir, f"features_{Path(image_path).stem}.png")
                self._plot_feature_maps(feature_maps, feature_viz_path)
                results['visualizations']['feature_maps'] = feature_viz_path
                print(f"💾 Feature maps saved: {feature_viz_path}")
        
        # Attention (for Vision Transformers)
        if show_attention:
            attention_maps = self.attention_viz.get_attention_maps(image_tensor)
            if attention_maps:
                attention_viz_path = os.path.join(save_dir, f"attention_{Path(image_path).stem}.png")
                self._plot_attention_maps(attention_maps, attention_viz_path)
                results['visualizations']['attention'] = attention_viz_path
                print(f"💾 Attention maps saved: {attention_viz_path}")
        
        return results
    
    def _create_gradcam_visualization(self, original_image, cam):
        """Create GradCAM overlay visualization."""
        # Resize CAM to match original image
        original_size = original_image.size
        cam_resized = cv2.resize(cam, original_size)
        
        # Normalize CAM
        cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy
        original_np = np.array(original_image)
        
        # Overlay heatmap on original image
        overlay = heatmap * 0.4 + original_np * 0.6
        overlay = np.uint8(overlay)
        
        return Image.fromarray(overlay)
    
    def _extract_key_feature_maps(self, image_tensor):
        """Extract feature maps from key layers."""
        # Find some key layers to visualize
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)
        
        if not conv_layers:
            return None
        
        # Select a few key layers (beginning, middle, end)
        key_layers = []
        if len(conv_layers) >= 3:
            key_layers = [conv_layers[0], conv_layers[len(conv_layers)//2], conv_layers[-1]]
        else:
            key_layers = conv_layers
        
        self.feature_viz.register_hooks(key_layers)
        
        feature_maps = {}
        for layer_name in key_layers:
            features = self.feature_viz.get_feature_maps(image_tensor, layer_name)
            if features is not None:
                feature_maps[layer_name] = features[0]  # Remove batch dimension
        
        self.feature_viz.remove_hooks()
        return feature_maps
    
    def _plot_feature_maps(self, feature_maps, save_path):
        """Plot feature maps in a grid."""
        n_layers = len(feature_maps)
        fig, axes = plt.subplots(n_layers, 8, figsize=(20, 3*n_layers))
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        for i, (layer_name, features) in enumerate(feature_maps.items()):
            # Show first 8 channels
            for j in range(min(8, features.shape[0])):
                ax = axes[i, j] if n_layers > 1 else axes[j]
                ax.imshow(features[j], cmap='viridis')
                ax.set_title(f'{layer_name}\nCh {j}', fontsize=8)
                ax.axis('off')
            
            # Hide unused subplots
            for j in range(features.shape[0], 8):
                if n_layers > 1:
                    axes[i, j].axis('off')
                else:
                    axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_maps(self, attention_maps, save_path):
        """Plot attention maps."""
        # This is a simplified version - actual implementation depends on model architecture
        n_maps = min(4, len(attention_maps))
        fig, axes = plt.subplots(1, n_maps, figsize=(4*n_maps, 4))
        if n_maps == 1:
            axes = [axes]
        
        for i, (name, attn_map) in enumerate(list(attention_maps.items())[:n_maps]):
            # Simplistic attention visualization
            if len(attn_map.shape) >= 3:
                # Average across heads and sequence length for visualization
                attn_avg = attn_map.mean(dim=0).mean(dim=0).cpu().numpy()
                axes[i].imshow(attn_avg, cmap='hot', interpolation='nearest')
                axes[i].set_title(f'Attention: {name}')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def analyze_confusion_cases(self, data_dir: str, save_dir: Optional[str] = None):
        """Analyze cases where model is confused (e.g., tomato diseases)."""
        # Use centralized visualization directory if no save_dir specified
        if save_dir is None:
            config_paths.ensure_directories()
            save_dir = os.path.join(config_paths.VISUALIZATIONS_DIR, "confusion_analysis")
        
        # Ensure save_dir is not None for type checker
        assert save_dir is not None
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"🔍 Starting confusion analysis...")
        print(f"📁 Results will be saved to: {save_dir}")
        
        # Focus on tomato disease classes
        tomato_classes = [i for i, name in enumerate(self.class_names) 
                         if 'tomato' in name.lower() and 'healthy' not in name.lower()]
        
        print(f"🍅 Found {len(tomato_classes)} tomato disease classes:")
        for i in tomato_classes:
            print(f"   {i}: {self.class_names[i]}")
        
        # TODO: Add implementation for batch processing of confusion cases
        # This would involve loading test images, finding misclassified ones,
        # and generating visualizations for analysis
        
    def cleanup(self):
        """Clean up all hooks and resources."""
        self.gradcam.remove_hooks()
        self.feature_viz.remove_hooks()
        self.attention_viz.remove_hooks()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='🔍 Model Visualization Toolkit')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, help='Single image to analyze')
    parser.add_argument('--data-dir', type=str, help='Directory of images to analyze')
    parser.add_argument('--save-dir', type=str, help='Directory to save visualizations (defaults to centralized visualizations directory)')
    parser.add_argument('--config', type=str, choices=['cnn', 'dinov2'], default='cnn', help='Config type')
    parser.add_argument('--gradcam', action='store_true', help='Generate GradCAM visualizations')
    parser.add_argument('--features', action='store_true', help='Generate feature map visualizations')
    parser.add_argument('--attention', action='store_true', help='Generate attention visualizations')
    parser.add_argument('--confusion', action='store_true', help='Analyze confusion cases')
    
    args = parser.parse_args()
    
    # Use centralized visualization directory if no save_dir specified
    save_dir = args.save_dir or config_paths.VISUALIZATIONS_DIR
    
    # Select config
    config_module = config_dinov2 if args.config == 'dinov2' else config
    
    # Initialize pipeline
    viz_pipeline = ModelVisualizationPipeline(args.model, config_module)
    
    try:
        if args.image:
            # Analyze single image
            results = viz_pipeline.analyze_single_image(
                args.image, save_dir, 
                show_gradcam=args.gradcam or not any([args.features, args.attention]),
                show_feature_maps=args.features,
                show_attention=args.attention
            )
            print(f"✅ Analysis complete! Results saved to {save_dir}")
            
        elif args.confusion:
            # Analyze confusion cases
            viz_pipeline.analyze_confusion_cases(args.data_dir, save_dir)
            
        else:
            print("❌ Please specify --image for single image analysis or --confusion for confusion analysis")
    
    finally:
        viz_pipeline.cleanup()


if __name__ == "__main__":
    main() 