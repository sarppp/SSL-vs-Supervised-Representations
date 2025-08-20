import os

# PROJECT_ROOT should be two levels up from src/config/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Data paths (still in project root)
CLEAN_DATASET_PICKLE = os.path.join(PROJECT_ROOT, "clean_dataset.pkl")
BASE_DATA_DIR = "/root/crop_pest_data"  # Correct path to dataset
#BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "crop_pest_data")

# ------------------------------------------------------------------
# Output hierarchy
#   outputs/
#       checkpoints/   ← saved weights & checkpoints
#       logs/          ← text / json logs
#       visualizations/← figures
# ------------------------------------------------------------------

# Base *outputs* folder lives in the project root
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Checkpoints (and legacy "models") directory live **inside** outputs
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")

# Back-compat alias – some code still expects ``MODELS_DIR``
MODELS_DIR = CHECKPOINTS_DIR

# Output paths (using new organized structure)
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, "visualizations")

# Additional output sub-folders
TRAINING_RESULTS_DIR = os.path.join(OUTPUTS_DIR, "training_results")
EVALUATION_RESULTS_DIR = os.path.join(OUTPUTS_DIR, "evaluation_results")
COMPARISON_RESULTS_DIR = os.path.join(OUTPUTS_DIR, "comparison_results")

# Visualization sub-folder for attention heatmaps
ATTENTION_VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, "attention_visualizations")

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
        OUTPUTS_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        VISUALIZATIONS_DIR,
        TRAINING_RESULTS_DIR,
        EVALUATION_RESULTS_DIR,
        COMPARISON_RESULTS_DIR,
        ATTENTION_VISUALIZATIONS_DIR
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