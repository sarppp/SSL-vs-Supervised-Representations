import os
import pickle
from PIL import Image
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
from ..config import config_paths

def validate_image(args):
    """Validate single image - for parallel processing"""
    path, label = args
    try:
        with Image.open(path) as img:
            img.verify()  # Verify image integrity
        # Re-open for RGB conversion check
        with Image.open(path) as img:
            img.convert('RGB')
        return path, label, True
    except Exception as e:
        return path, label, False

def clean_and_save_dataset(data_dir, save_path, max_workers=None, use_relative_paths=True):
    """Clean dataset using configurable CPU cores and save clean version with summary."""
    
    # Use optimal CPU cores (leave 1-2 cores free for system)
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)
    
    print(f"🔍 Scanning dataset directory: {data_dir}")
    
    # Collect original data
    original_data = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    for class_name in sorted(os.listdir(data_dir)):  # Sort for consistency
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            class_images = []
            for image_name in os.listdir(class_dir):
                if any(image_name.lower().endswith(ext) for ext in valid_extensions):
                    image_path = os.path.join(class_dir, image_name)
                    if os.path.isfile(image_path):
                        class_images.append((image_path, class_name))
            
            # Sort images within each class for consistency
            class_images.sort()
            original_data.extend(class_images)

    print(f"📊 Found: {len(original_data)} images across {len(set(label for _, label in original_data))} classes")
    print(f"🚀 Using {max_workers} CPU cores for validation...")
    
    # Parallel validation with progress bar
    clean_paths, clean_labels = [], []
    corrupted_files = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(validate_image, original_data), 
            total=len(original_data),
            desc="🔍 Validating images",
            unit="img",
            ncols=80
        ))
    
    for path, label, is_valid in results:
        if is_valid:
            # Convert to relative paths if requested
            if use_relative_paths:
                rel_path = os.path.relpath(path, data_dir)
                clean_paths.append(rel_path)
            else:
                clean_paths.append(path)
            clean_labels.append(label)
        else:
            corrupted_files.append(path)

    print(f"\n✅ Clean: {len(clean_paths)} images")
    print(f"🗑️ Removed: {len(corrupted_files)} corrupted images")
    
    if corrupted_files:
        print(f"📝 Corrupted files:")
        for file in corrupted_files[:5]:  # Show first 5
            print(f"  - {file}")
        if len(corrupted_files) > 5:
            print(f"  ... and {len(corrupted_files) - 5} more")
    
    # Show class distribution
    distribution = Counter(clean_labels)
    print(f"\n📋 Class distribution ({len(distribution)} classes):")
    for class_name, count in sorted(distribution.items()):
        print(f"  {class_name}: {count:,} images")
    
    # Calculate dataset statistics
    total_clean = len(clean_paths)
    avg_per_class = total_clean / len(distribution)
    min_class = min(distribution.values())
    max_class = max(distribution.values())
    
    print(f"\n📈 Dataset statistics:")
    print(f"  Average per class: {avg_per_class:.1f}")
    print(f"  Min class size: {min_class:,}")
    print(f"  Max class size: {max_class:,}")
    print(f"  Imbalance ratio: {max_class/min_class:.2f}:1")
    
    # Save clean dataset with metadata
    dataset_info = {
        'paths': clean_paths,
        'labels': clean_labels,
        'data_dir': data_dir,
        'use_relative_paths': use_relative_paths,
        'total_images': len(clean_paths),
        'num_classes': len(distribution),
        'class_distribution': dict(distribution),
        'corrupted_count': len(corrupted_files)
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataset_info, f)
    
    file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
    print(f"\n💾 Clean dataset saved to: {save_path} ({file_size:.2f} MB)")
    
    return clean_paths, clean_labels

if __name__ == "__main__":
    # Example usage
    data_dir = config_paths.BASE_DATA_DIR  # Use centralized data dir
    save_path = config_paths.CLEAN_DATASET_PICKLE
    clean_paths, clean_labels = clean_and_save_dataset(data_dir, save_path)