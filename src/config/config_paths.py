import os

# PROJECT_ROOT should be two levels up from src/config/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Data paths (still in project root)
CLEAN_DATASET_PICKLE = os.path.join(PROJECT_ROOT, "clean_dataset.pkl")
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "crop_pest_data")

# Model paths (using new organized structure)
MODELS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")  # Renamed for clarity
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# Output paths (using new organized structure)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "visualizations")

# Source directories
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(SRC_DIR, "data")
MODELS_SRC_DIR = os.path.join(SRC_DIR, "models")
TRAINING_DIR = os.path.join(SRC_DIR, "training")
EVALUATION_DIR = os.path.join(SRC_DIR, "evaluation")
UTILS_DIR = os.path.join(SRC_DIR, "utils")
VISUALIZATION_SRC_DIR = os.path.join(SRC_DIR, "visualization")
CONFIG_DIR = os.path.join(SRC_DIR, "config")

# Legacy support for old thesis_codes references (now points to src)
THESIS_CODES_DIR = SRC_DIR

# Ensure directories exist
def ensure_directories():
    """Create directories if they don't exist."""
    dirs_to_create = [
        CHECKPOINTS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        VISUALIZATIONS_DIR
    ]
    
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)

# Utility functions
def get_model_path(model_name):
    """Get path for a model file."""
    return os.path.join(CHECKPOINTS_DIR, f"{model_name}.pth")

def get_checkpoint_path(model_name, epoch=None):
    """Get path for a checkpoint file."""
    if epoch is not None:
        return os.path.join(CHECKPOINTS_DIR, f"{model_name}_epoch_{epoch}.pth")
    return os.path.join(CHECKPOINTS_DIR, f"{model_name}_best.pth")

def get_log_path(log_name):
    """Get path for a log file."""
    return os.path.join(LOGS_DIR, f"{log_name}.log")

def get_visualization_path(viz_name):
    """Get path for a visualization file."""
    return os.path.join(VISUALIZATIONS_DIR, f"{viz_name}.png")