#!/usr/bin/env python3
"""
🔍 Model Inference Visualization
Visualize what your trained models learned by testing on random images
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import sys
import random
from pathlib import Path
import json
from torchvision import transforms
import torch.nn.functional as F

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_codes_dir = os.path.join(current_dir, 'thesis_codes')

if current_dir not in sys.path:
    sys.path.append(current_dir)
if thesis_codes_dir not in sys.path:
    sys.path.append(thesis_codes_dir)

try:
    import config
    import config_dinov2
    import model_setup
    import data_splitter
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class ModelInferenceVisualizer:
    """Visualize model predictions and learned patterns."""
    
    def __init__(self, model_path, config_module):
        self.model_path = model_path
        self.config_module = config_module
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and metadata
        self.model, self.class_names, self.model_info = self._load_model()
        self.model.eval()
        
        # Setup transforms
        self.transform = self._setup_transforms()
        
        print(f"✅ Model loaded: {self.model_info['model_name']}")
        print(f"📊 Classes: {len(self.class_names)}")
        print(f"🎯 Training accuracy: {self.model_info.get('val_acc', 'N/A'):.2f}%")
        print(f"📐 Image size: {self.config_module.IMAGE_SIZE}")
    
    def _load_model(self):
        """Load trained model and extract metadata."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model info
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        model_name = checkpoint.get('model_name', self.config_module.MODEL_NAME)
        
        # Create model
        model = model_setup.create_model(num_classes, model_name, self.config_module)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        # Extract training info
        model_info = {
            'model_name': model_name,
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_acc': checkpoint.get('val_acc', 0),
            'train_loss': checkpoint.get('train_loss', 'N/A'),
            'val_loss': checkpoint.get('val_loss', 'N/A'),
            'timestamp': checkpoint.get('timestamp', 'N/A'),
            'batch_size': checkpoint.get('batch_size', 'N/A'),
            'optimizer_name': checkpoint.get('optimizer_name', 'N/A')
        }
        
        return model, class_names, model_info
    
    def _setup_transforms(self):
        """Setup image transforms matching training."""
        return transforms.Compose([
            transforms.Resize(self.config_module.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict_single_image(self, image_path):
        """Get model prediction for a single image."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
            predicted_class_idx = outputs.argmax(dim=1).item()
            confidence = probabilities[predicted_class_idx]
        
        # Get top 5 predictions
        top5_indices = probabilities.argsort()[-5:][::-1]
        top5_predictions = [(self.class_names[i], probabilities[i]) for i in top5_indices]
        
        return {
            'image_path': image_path,
            'original_image': image,
            'predicted_class': self.class_names[predicted_class_idx],
            'predicted_idx': predicted_class_idx,
            'confidence': confidence,
            'probabilities': probabilities,
            'top5_predictions': top5_predictions
        }
    
    def visualize_predictions(self, image_paths, save_path=None):
        """Visualize predictions for multiple images."""
        n_images = len(image_paths)
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        if n_images == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'{self.model_info["model_name"]} - Model Predictions', fontsize=16, fontweight='bold')
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single_image(image_path)
                
                # Top plot: Original image with prediction
                axes[0, i].imshow(result['original_image'])
                axes[0, i].set_title(f'Predicted: {result["predicted_class"]}\n'
                                   f'Confidence: {result["confidence"]:.3f}', 
                                   fontsize=10, fontweight='bold')
                axes[0, i].axis('off')
                
                # Bottom plot: Top 5 predictions bar chart
                top5_classes = [pred[0] for pred in result['top5_predictions']]
                top5_probs = [pred[1] for pred in result['top5_predictions']]
                
                # Truncate long class names
                top5_classes_short = [cls.replace(' ', '\n') if len(cls) > 15 else cls for cls in top5_classes]
                
                bars = axes[1, i].barh(range(5), top5_probs, color=plt.cm.viridis(np.linspace(0, 1, 5)))
                axes[1, i].set_yticks(range(5))
                axes[1, i].set_yticklabels(top5_classes_short, fontsize=8)
                axes[1, i].set_xlabel('Probability', fontsize=10)
                axes[1, i].set_title(f'Top 5 Predictions', fontsize=10)
                axes[1, i].set_xlim(0, 1)
                
                # Add probability values on bars
                for j, prob in enumerate(top5_probs):
                    axes[1, i].text(prob + 0.01, j, f'{prob:.3f}', 
                                   va='center', fontsize=8)
                
                print(f"📸 {Path(image_path).name}: {result['predicted_class']} ({result['confidence']:.3f})")
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                axes[0, i].text(0.5, 0.5, f'Error loading\n{Path(image_path).name}', 
                              ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].axis('off')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved: {save_path}")
        
        plt.show()
        return fig
    
    def compare_models_on_image(self, other_visualizer, image_path, save_path=None):
        """Compare predictions from two models on the same image."""
        result1 = self.predict_single_image(image_path)
        result2 = other_visualizer.predict_single_image(image_path)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Original image (shared)
        axes[0, 0].imshow(result1['original_image'])
        axes[0, 0].set_title(f'Original Image\n{Path(image_path).name}', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Model 1 prediction
        axes[0, 1].imshow(result1['original_image'])
        axes[0, 1].set_title(f'{self.model_info["model_name"]}\n'
                           f'Predicted: {result1["predicted_class"]}\n'
                           f'Confidence: {result1["confidence"]:.3f}', 
                           fontweight='bold')
        axes[0, 1].axis('off')
        
        # Model 2 prediction  
        axes[0, 2].imshow(result2['original_image'])
        axes[0, 2].set_title(f'{other_visualizer.model_info["model_name"]}\n'
                           f'Predicted: {result2["predicted_class"]}\n'
                           f'Confidence: {result2["confidence"]:.3f}', 
                           fontweight='bold')
        axes[0, 2].axis('off')
        
        # Model 1 top 5
        top5_classes1 = [pred[0] for pred in result1['top5_predictions']]
        top5_probs1 = [pred[1] for pred in result1['top5_predictions']]
        top5_classes1_short = [cls.replace(' ', '\n') if len(cls) > 15 else cls for cls in top5_classes1]
        
        axes[1, 1].barh(range(5), top5_probs1, color='skyblue')
        axes[1, 1].set_yticks(range(5))
        axes[1, 1].set_yticklabels(top5_classes1_short, fontsize=9)
        axes[1, 1].set_xlabel('Probability')
        axes[1, 1].set_title(f'{self.model_info["model_name"]} - Top 5')
        axes[1, 1].set_xlim(0, 1)
        
        # Model 2 top 5
        top5_classes2 = [pred[0] for pred in result2['top5_predictions']]
        top5_probs2 = [pred[1] for pred in result2['top5_predictions']]
        top5_classes2_short = [cls.replace(' ', '\n') if len(cls) > 15 else cls for cls in top5_classes2]
        
        axes[1, 2].barh(range(5), top5_probs2, color='lightcoral')
        axes[1, 2].set_yticks(range(5))
        axes[1, 2].set_yticklabels(top5_classes2_short, fontsize=9)
        axes[1, 2].set_xlabel('Probability')
        axes[1, 2].set_title(f'{other_visualizer.model_info["model_name"]} - Top 5')
        axes[1, 2].set_xlim(0, 1)
        
        # Hide the bottom left subplot
        axes[1, 0].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Comparison saved: {save_path}")
        
        plt.show()
        
        # Print comparison summary
        print(f"\n🔍 MODEL COMPARISON ON {Path(image_path).name}:")
        print(f"📊 {self.model_info['model_name']:25} | {result1['predicted_class']:25} | {result1['confidence']:.3f}")
        print(f"📊 {other_visualizer.model_info['model_name']:25} | {result2['predicted_class']:25} | {result2['confidence']:.3f}")
        
        if result1['predicted_class'] == result2['predicted_class']:
            print("✅ Both models agree!")
        else:
            print("❌ Models disagree!")
        
        return fig, result1, result2

def get_random_images_from_dataset(base_dir='crop_pest_data', n_images=4, random_seed=42):
    """Get random images from the dataset."""
    random.seed(random_seed)
    
    all_images = []
    for class_dir in Path(base_dir).iterdir():
        if class_dir.is_dir():
            class_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            all_images.extend(class_images)
    
    if len(all_images) < n_images:
        print(f"⚠️  Only found {len(all_images)} images, using all of them")
        return all_images
    
    return random.sample(all_images, n_images)

def main():
    """Main function to visualize model predictions."""
    print("🚀 Model Inference Visualization")
    print("=" * 50)
    
    # Model paths (update these to match your actual model files)
    cnn_model_path = 'models/best_cnn_b4_acc73.40_20250702_133941.pth'
    dinov2_model_path = 'models/best_dino_vits14_acc74.47_20250702_134026.pth'
    
    # Check if models exist
    if not os.path.exists(cnn_model_path):
        print(f"❌ CNN model not found: {cnn_model_path}")
        return
    if not os.path.exists(dinov2_model_path):
        print(f"❌ DINOv2 model not found: {dinov2_model_path}")
        return
    
    print(f"🔍 Loading models...")
    
    # Initialize visualizers
    cnn_viz = ModelInferenceVisualizer(cnn_model_path, config)
    dinov2_viz = ModelInferenceVisualizer(dinov2_model_path, config_dinov2)
    
    print(f"\n📂 Getting random images from dataset...")
    random_images = get_random_images_from_dataset('crop_pest_data', n_images=4)
    
    if not random_images:
        print("❌ No images found in dataset!")
        return
    
    print(f"🎯 Found {len(random_images)} random images")
    for img in random_images:
        print(f"   📸 {img}")
    
    # Create output directory
    os.makedirs('model_visualizations', exist_ok=True)
    
    print(f"\n🔍 Visualizing CNN predictions...")
    cnn_viz.visualize_predictions(
        [str(img) for img in random_images], 
        save_path='model_visualizations/cnn_predictions.png'
    )
    
    print(f"\n🔍 Visualizing DINOv2 predictions...")
    dinov2_viz.visualize_predictions(
        [str(img) for img in random_images], 
        save_path='model_visualizations/dinov2_predictions.png'
    )
    
    print(f"\n🔍 Comparing models on individual images...")
    for i, img_path in enumerate(random_images[:2]):  # Compare on first 2 images
        print(f"\n🎯 Comparing on image {i+1}: {Path(img_path).name}")
        cnn_viz.compare_models_on_image(
            dinov2_viz, 
            str(img_path),
            save_path=f'model_visualizations/comparison_image_{i+1}.png'
        )
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 Check 'model_visualizations/' directory for saved plots")

if __name__ == "__main__":
    main() 