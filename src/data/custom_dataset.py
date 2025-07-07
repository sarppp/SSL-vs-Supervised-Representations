import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import time
import hashlib
import threading
from io import BytesIO

class CustomCropDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, fallback_size=(224, 224), validate_images=True,
                 class_to_idx: dict | None = None, ignore_index: int = -1):
        """A custom dataset that supports *few-shot label hiding* by allowing an ``ignore_index`` (default ``-1``)
        to denote *unlabelled* samples **without** treating that value as its own class.  

        When a common ``class_to_idx`` mapping is provided, it is used **as-is** so that *all* splits
        (train/val/test) share the exact same label → index mapping.  If not provided, the mapping is
        built from the labels list **excluding** the ``ignore_index`` value.
        """

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.fallback_size = fallback_size
        self.validate_images = validate_images
        self.ignore_index = ignore_index

        # Thread lock for file operations (prevents PIL IO race conditions on multi-worker dataloaders)
        self._file_lock = threading.Lock()

        # Cache for validated images so we don't repeatedly hit the disk
        self._validated_cache = {}

        # ------------------------------------------------------------------
        # Build / assign the class mapping *once* so that every dataset split
        # is guaranteed to use the same indices, and the special ``ignore_index``
        # **never** becomes a legitimate class.
        # ------------------------------------------------------------------
        if class_to_idx is None:
            # Derive mapping from the observed labels, skipping the ignore_index
            unique_labels = sorted(list({lbl for lbl in labels if lbl != ignore_index}))
            self.classes = unique_labels
            self.class_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        else:
            # Re-use supplied mapping (order matters!)
            self.class_to_idx = class_to_idx
            # Preserve a reproducible ordered list of class names/ids for external use
            # (e.g. for "dataset.classes" export)
            # Sort by the mapped index to keep deterministic ordering
            self.classes = [label for label, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

        # Finally, convert raw labels to integer targets **but** keep ``ignore_index``
        # untouched so that loss functions such as ``nn.CrossEntropyLoss`` with
        # ``ignore_index`` can properly skip them.
        self.targets = [ignore_index if lbl == ignore_index else self.class_to_idx[lbl] for lbl in labels]
        
        # 🔍 Pre-validate images during initialization to catch corruption early
        if self.validate_images:
            print("🔍 Pre-validating images to prevent runtime corruption...")
            self._prevalidate_images()

    def _get_file_hash(self, filepath):
        """Get MD5 hash of file for corruption detection"""
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except:
            return None

    def _is_valid_image(self, img_path):
        """Validate if image file is not corrupted"""
        try:
            # Check file exists and has size
            if not os.path.exists(img_path):
                return False, "File does not exist"
                
            file_size = os.path.getsize(img_path)
            if file_size == 0:
                return False, "File is empty (0 bytes)"
            
            if file_size < 100:  # Very small files are likely corrupted
                return False, f"File too small ({file_size} bytes)"
            
            # Try to open and verify the image
            with Image.open(img_path) as img:
                img.verify()
            
            # If verify passes, try to actually load it
            with Image.open(img_path) as img:
                img.load()
                img.convert('RGB')
            
            return True, "Valid"
            
        except Exception as e:
            return False, str(e)

    def _prevalidate_images(self):
        """Pre-validate all images and store results"""
        corrupted_files = []
        
        for idx, img_path in enumerate(self.image_paths):
            if idx % 500 == 0:
                print(f"   Validating images: {idx}/{len(self.image_paths)}")
            
            is_valid, error_msg = self._is_valid_image(img_path)
            self._validated_cache[img_path] = is_valid
            
            if not is_valid:
                corrupted_files.append((img_path, error_msg))
        
        if corrupted_files:
            print(f"⚠️  Found {len(corrupted_files)} corrupted files:")
            for filepath, error in corrupted_files[:10]:  # Show first 10
                print(f"   {filepath}: {error}")
            if len(corrupted_files) > 10:
                print(f"   ... and {len(corrupted_files) - 10} more")
            
            # Save corrupted files list for manual cleanup
            corrupted_list_file = "corrupted_files_detected.txt"
            with open(corrupted_list_file, 'w') as f:
                for filepath, error in corrupted_files:
                    f.write(f"{filepath}\t{error}\n")
            print(f"💾 Corrupted files list saved to: {corrupted_list_file}")
        else:
            print("✅ All images validated successfully!")

    def __len__(self):
        return len(self.image_paths)

    def _load_image_safely(self, img_path):
        """Load image with multiple safety mechanisms"""
        
        # Use thread lock to prevent concurrent access issues
        with self._file_lock:
            # Check cache first
            if self.validate_images and img_path in self._validated_cache:
                if not self._validated_cache[img_path]:
                    raise ValueError(f"Image pre-validated as corrupted: {img_path}")
            
            # Method 1: Standard PIL loading with explicit resource management
            try:
                with Image.open(img_path) as img_temp:
                    # Copy image data to memory to avoid file handle issues
                    img = img_temp.copy().convert('RGB')
                    return img
            except Exception as e1:
                pass
            
            # Method 2: Load into memory first, then parse
            try:
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                
                # Validate bytes
                if len(img_bytes) == 0:
                    raise ValueError("File contains no data")
                
                # Check for JPEG signature
                if not img_bytes.startswith(b'\xff\xd8'):
                    raise ValueError("File does not have JPEG signature")
                
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
                return img
                
            except Exception as e2:
                pass
            
            # Method 3: Try to repair/reload
            try:
                # Force re-read with explicit verification
                with Image.open(img_path) as img_temp:
                    img_temp.verify()
                
                # Re-open after verify (verify closes the file)
                with Image.open(img_path) as img_temp:
                    img = img_temp.copy().convert('RGB')
                    return img
                    
            except Exception as e3:
                # All methods failed
                raise RuntimeError(f"All loading methods failed for {img_path}. Last error: {e3}")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.targets[idx]

        try:
            # Load image safely
            img = self._load_image_safely(img_path)
            
        except Exception as e:
            print(f"🚨 ERROR: Failed to load {img_path}: {e}")
            
            # Log corruption for investigation
            with open("runtime_corruptions.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{img_path}\t{str(e)}\n")
            
            # Return black image fallback
            img = Image.new('RGB', self.fallback_size, (0, 0, 0))
            print(f"🖤 Using black image fallback for {img_path}")

        # Apply transforms
        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                print(f"🚨 Transform error for {img_path}: {e}")
                # Create black tensor fallback
                channels = 3
                height, width = self.fallback_size
                img = torch.zeros(channels, height, width)

        return img, label

    def get_corruption_stats(self):
        """Get statistics about corrupted files"""
        if not self.validate_images:
            return "Validation not enabled"
        
        total_files = len(self._validated_cache)
        corrupted_files = sum(1 for valid in self._validated_cache.values() if not valid)
        
        return {
            'total_files': total_files,
            'corrupted_files': corrupted_files,
            'corruption_rate': corrupted_files / total_files * 100 if total_files > 0 else 0
        }
