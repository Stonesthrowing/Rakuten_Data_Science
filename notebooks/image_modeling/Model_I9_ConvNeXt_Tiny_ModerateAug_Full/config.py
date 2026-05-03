# ============================================================
# config.py
# All hyperparameters and paths for:
#   Model_I9_ConvNeXt_Tiny_ModerateAug_Full
#
# Key differences vs. ResNet50 models:
#   - Architecture: ConvNeXt-Tiny (stronger than ResNet50)
#   - Mixed Precision Training (AMP) enabled via GradScaler
#   - Differential LR: backbone (1e-5) vs. head (1e-4)
#   - Smaller batch size (32) due to ConvNeXt memory footprint
#   - TrivialAugmentWide augmentation strategy
#   - WeightDecay = 0.05 (higher, suits ConvNeXt)
#
# To start a new experiment, copy this file, adjust values,
# and pass the new config to train.py.
# ============================================================

from pathlib import Path

# ------------------------------------------------------------
# Experiment identity
# ------------------------------------------------------------
RUN_NAME = "I9_ConvNeXt_Tiny_ModerateAug_Full"

# ------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32           # ConvNeXt is more memory-intensive than ResNet50
NUM_WORKERS = 2

MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 6

# Differential learning rates: lower for backbone to preserve pretrained weights
LR_BACKBONE = 1e-5
LR_HEAD     = 1e-4
WEIGHT_DECAY = 0.05       # Higher than ResNet models, suits ConvNeXt

LABEL_SMOOTHING = 0.1
DROPOUT = 0.4             # Applied in ConvNeXt classification head

# Scheduler
SCHEDULER_MODE    = "max"
SCHEDULER_FACTOR  = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR  = 1e-7

# Mixed Precision Training (AMP) – set False if GPU does not support it
USE_AMP = True

# ------------------------------------------------------------
# Resume training
# ------------------------------------------------------------
RESUME_TRAINING   = False
CHECKPOINT_SOURCE = "local_last"   # "local_last" | "local_best"

# ------------------------------------------------------------
# Project directory structure  –  adjust PROJECT_DIR to your machine
# ------------------------------------------------------------
PROJECT_DIR = Path(
    r"C:\Users\felix\Documents\DS_MLE_MasterSystem\06_PROJECTS\Project_01_Rakuten_Multimodal"
)

DATA_DIR        = PROJECT_DIR / "data"
RAW_IMG_DIR     = DATA_DIR   / "raw" / "images"

OUTPUT_DIR      = PROJECT_DIR / "outputs"
FIGURE_DIR      = PROJECT_DIR / "figures"
MODEL_DIR       = PROJECT_DIR / "models"

SPLIT_DIR             = OUTPUT_DIR  / "image_modeling"
LOCAL_IMAGE_TRAIN_DIR = RAW_IMG_DIR / "image_train"
LOCAL_IMAGE_TEST_DIR  = RAW_IMG_DIR / "image_test"

# Run-specific output folders (auto-created by train.py)
LOCAL_OUTPUT_ROOT = OUTPUT_DIR / RUN_NAME
LOCAL_MODEL_ROOT  = MODEL_DIR  / RUN_NAME
LOCAL_FIG_ROOT    = FIGURE_DIR / RUN_NAME

# Checkpoint & history file paths (derived, do not change)
LAST_CKPT_LOCAL    = LOCAL_MODEL_ROOT  / "last_checkpoint.pt"
BEST_CKPT_LOCAL    = LOCAL_MODEL_ROOT  / "best_checkpoint.pt"
BEST_WEIGHTS_LOCAL = LOCAL_MODEL_ROOT  / "best_model_state_dict.pt"
HISTORY_JSON_LOCAL = LOCAL_OUTPUT_ROOT / "history.json"
METADATA_JSON_LOCAL = LOCAL_OUTPUT_ROOT / "run_metadata.json"
