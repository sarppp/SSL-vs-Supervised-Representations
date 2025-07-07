"""
Crop Pest Detection Research Framework
====================================

This package provides a comprehensive framework for crop pest detection using
deep learning models including CNNs and Vision Transformers.

Directory Structure:
- src/config/: Configuration files and constants
- src/data/: Data handling, loading, and preprocessing
- src/models/: Model architectures and setup
- src/training/: Training loops and utilities
- src/evaluation/: Model evaluation and metrics
- src/utils/: Utility functions (logging, class weights, etc.)
- src/visualization/: Visualization and analysis tools
"""

import os
import sys

# Add all src subdirectories to Python path for easy imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_subdirs = ['config', 'data', 'models', 'training', 'evaluation', 'utils', 'visualization']

for _subdir in _subdirs:
    _path = os.path.join(_current_dir, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Also add the main src directory
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Add the project root to path (parent of src)
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
