# ============================================================
# config.py
# All hyperparameters and paths for:
#   Model_I6_ResNet50_ModerateAug_Partial
#
# To start a new experiment, copy this file, adjust the values
# below, and pass the new config to train.py.
# ============================================================

from pathlib import Path

# ------------------------------------------------------------
# Experiment identity
# ------------------------------------------------------------
RUN_NAME = "I6_ResNet50_ModerateAug_Partial"

# ------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 128          # Good default for GPU VRAM
NUM_WORKERS = 2

MAX_EPOCHS = 18
EARLY_STOPPING_PATIENCE = 6

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# Label smoothing: 0.0 = off (E2 style), 0.1 = on (E3 style)
LABEL_SMOOTHING = 0.1

# Dropout rate in the classification head
DROPOUT = 0.5

# Scheduler
SCHEDULER_MODE = "max"    # ReduceLROnPlateau monitors val macro-F1
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR = 1e-6

# ------------------------------------------------------------
# Resume training
# ------------------------------------------------------------
RESUME_TRAINING = False
CHECKPOINT_SOURCE = "local_last"   # "local_last" | "local_best"

# ------------------------------------------------------------
# Project directory structure  –  adjust PROJECT_DIR to your machine
# ------------------------------------------------------------
PROJECT_DIR = Path(
    r"C:\Users\felix\Documents\DS_MLE_MasterSystem\06_PROJECTS\Project_01_Rakuten_Multimodal"
)

DATA_DIR        = PROJECT_DIR / "data"
RAW_DIR         = DATA_DIR   / "raw"
RAW_IMG_DIR     = RAW_DIR    / "images"

OUTPUT_DIR      = PROJECT_DIR / "outputs"
FIGURE_DIR      = PROJECT_DIR / "figures"
MODEL_DIR       = PROJECT_DIR / "models"

SPLIT_DIR            = OUTPUT_DIR  / "image_modeling"
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
