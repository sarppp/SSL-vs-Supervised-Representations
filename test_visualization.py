#!/usr/bin/env python3
"""
🔍 Quick Model Visualization Test
===============================
Test script to visualize your trained CNN model behavior
"""

import os
import sys
import glob
from pathlib import Path

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
thesis_codes_dir = os.path.join(current_dir, 'thesis_codes')
if thesis_codes_dir not in sys.path:
    sys.path.append(thesis_codes_dir)

# Import our visualization module
try:
    from model_visualization import ModelVisualizationPipeline
    print("✅ Model visualization module imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def find_model_file():
    """Find the model file."""
    possible_paths = [
        "models/best_cnn_b3_acc82.32_20250702_061634.pth",
        "./models/best_cnn_b3_acc82.32_20250702_061634.pth",
        "best_cnn_b3_acc82.32_20250702_061634.pth",
        "./best_cnn_b3_acc82.32_20250702_061634.pth"
    ]
    
    # Also search for any .pth files
    for pattern in ["**/*.pth", "*.pth", "models/*.pth"]:
        found_files = glob.glob(pattern, recursive=True)
        if found_files:
            print(f"🔍 Found model files: {found_files}")
            # Look for the specific one you mentioned
            for f in found_files:
                if "best_cnn_b3_acc82.32" in f:
                    return f
            # Return the first one found
            return found_files[0]
    
    # Try the specific paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def find_test_images():
    """Find some test images to analyze."""
    image_patterns = [
        "crop_pest_data/**/*tomato*.jpg",
        "crop_pest_data/**/*tomato*.png", 
        "crop_pest_data/**/*.jpg",
        "crop_pest_data/**/*.png",
        "**/*tomato*.jpg",
        "**/*tomato*.png",
        "**/*.jpg",
        "**/*.png"
    ]
    
    test_images = []
    for pattern in image_patterns:
        found = glob.glob(pattern, recursive=True)
        test_images.extend(found[:5])  # Limit to 5 images per pattern
        if len(test_images) >= 3:  # We just need a few for testing
            break
    
    return test_images[:3]  # Return max 3 images

def main():
    """Main visualization test."""
    print("🔍 TESTING MODEL VISUALIZATION")
    print("=" * 50)
    
    # Find model
    model_path = find_model_file()
    if not model_path:
        print("❌ Could not find your model file!")
        print("   Please make sure the model file exists:")
        print("   models/best_cnn_b3_acc82.32_20250702_061634.pth")
        return
    
    print(f"✅ Found model: {model_path}")
    
    # Find test images
    test_images = find_test_images()
    if not test_images:
        print("❌ Could not find any test images!")
        print("   Please make sure you have images in crop_pest_data/")
        return
    
    print(f"✅ Found {len(test_images)} test images:")
    for img in test_images:
        print(f"   📸 {img}")
    
    # Initialize visualization pipeline
    try:
        print(f"\n🚀 Initializing visualization pipeline...")
        viz_pipeline = ModelVisualizationPipeline(model_path)
        print(f"✅ Pipeline initialized successfully!")
        
        # Test on first image
        test_image = test_images[0]
        print(f"\n🔍 Analyzing: {test_image}")
        
        # Create output directory
        save_dir = "visualization_results"
        os.makedirs(save_dir, exist_ok=True)
        
        # Run analysis
        results = viz_pipeline.analyze_single_image(
            test_image, 
            save_dir=save_dir,
            show_gradcam=True,
            show_feature_maps=True,
            show_attention=False  # CNN doesn't have attention
        )
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"🎯 Predicted: {results['predictions']['predicted_class']}")
        print(f"🔥 Confidence: {results['predictions']['confidence']:.3f}")
        print(f"🏆 Top 5 predictions:")
        for i, (class_name, prob) in enumerate(results['predictions']['top_5']):
            print(f"   {i+1}. {class_name}: {prob:.3f}")
        
        print(f"\n💾 Visualizations saved to: {save_dir}/")
        print(f"   - GradCAM: Shows what the model focuses on")
        print(f"   - Feature Maps: Shows how the model processes the image")
        
        # Clean up
        viz_pipeline.cleanup()
        
        print(f"\n✅ Visualization test completed successfully!")
        print(f"🔍 Check the '{save_dir}' folder for your visualizations!")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 