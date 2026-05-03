# ============================================================
# config.py
# All paths and hyperparameters for:
#   MM_CamemBERT_ConvNeXtBase_IntermediateFusion
#
# This model trains a lightweight MLP fusion head on top of
# pre-extracted features from:
#   - Model_I12_ConvNeXt_Base_ModerateAug_Full  → 1024d features
#   - Model_T8_CamemBERT_FullFineTune_L128      →  768d features
# Concatenated input: 1024 + 768 = 1792 dimensions.
#
# Run order:
#   1. train.py  (I12) → exports train/val_features_1024d.npy
#   2. train.py  (T8)  → exports text_train/val_features_768d.npy
#   3. train.py  (this model)
#   4. evaluate.py
#   5. analyze_errors.py  (optional: top-5 error analysis)
#   6. gradcam.py         (optional: Grad-CAM visualisation)
# ============================================================

from pathlib import Path

# ------------------------------------------------------------
# Experiment identity
# ------------------------------------------------------------
RUN_NAME = "MM_CamemBERT_ConvNeXtBase_IntermediateFusion"

# ------------------------------------------------------------
# Model architecture
# ------------------------------------------------------------
IMG_FEATURE_DIM  = 1024   # ConvNeXt-Base avgpool output
TXT_FEATURE_DIM  =  768   # CamemBERT CLS-token
INPUT_DIM        = IMG_FEATURE_DIM + TXT_FEATURE_DIM   # 1792
NUM_CLASSES      = 27

# Fusion MLP head (V2 architecture – better regularised)
HIDDEN_DIM_1  = 256
HIDDEN_DIM_2  = 128
DROPOUT_1     = 0.5
DROPOUT_2     = 0.4

# ------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------
SEED       = 42
BATCH_SIZE = 64
NUM_WORKERS = 0       # Feature tensors fit in RAM; 0 workers is fastest

MAX_EPOCHS              = 30   # Early stopping will terminate before this
EARLY_STOPPING_PATIENCE =  3

LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 0.05   # Strong regularisation (V2 finding)

LABEL_SMOOTHING = 0.0   # Fusion head is lightweight; smoothing not needed

# ------------------------------------------------------------
# Grad-CAM analysis (used by gradcam.py)
# ------------------------------------------------------------
# Index into val_split.csv for the sample to analyse.
# Update this after running analyze_errors.py.
GRADCAM_TARGET_INDEX = 4921

# ------------------------------------------------------------
# Project directory structure  –  adjust PROJECT_DIR to your machine
# ------------------------------------------------------------
PROJECT_DIR = Path(
    r"C:\Users\felix\Documents\DS_MLE_MasterSystem\06_PROJECTS\Project_01_Rakuten_Multimodal"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
MODEL_DIR  = PROJECT_DIR / "models"
SPLIT_DIR  = OUTPUT_DIR  / "image_modeling"
IMAGE_DIR  = PROJECT_DIR / "data" / "raw" / "images" / "image_train"

# Source feature directories
IMAGE_MODEL_OUTPUT_DIR = OUTPUT_DIR / "I12_ConvNeXt_Base_ModerateAug_Full"
TEXT_MODEL_OUTPUT_DIR  = OUTPUT_DIR / "T8_CamemBERT_FullFineTune_L128"

# Image model weights (needed for Grad-CAM)
IMAGE_MODEL_WEIGHTS = MODEL_DIR / "I12_ConvNeXt_Base_ModerateAug_Full" / "best_model_state_dict.pt"

# Input feature files (produced by I12 and T8 feature export)
TRAIN_FEATURES_IMAGE = IMAGE_MODEL_OUTPUT_DIR / "train_features_1024d.npy"
TRAIN_FEATURES_TEXT  = TEXT_MODEL_OUTPUT_DIR  / "text_train_features_768d.npy"
VAL_FEATURES_IMAGE   = IMAGE_MODEL_OUTPUT_DIR / "val_features_1024d.npy"
VAL_FEATURES_TEXT    = TEXT_MODEL_OUTPUT_DIR  / "text_val_features_768d.npy"

# Run-specific output folders (auto-created by train.py)
LOCAL_OUTPUT_ROOT = OUTPUT_DIR / RUN_NAME
LOCAL_MODEL_ROOT  = MODEL_DIR  / RUN_NAME
LOCAL_FIG_ROOT    = FIGURE_DIR / RUN_NAME

# Output file paths
BEST_WEIGHTS_LOCAL  = LOCAL_MODEL_ROOT  / "best_fusion_model.pt"
HISTORY_JSON_LOCAL  = LOCAL_OUTPUT_ROOT / "history.json"
METADATA_JSON_LOCAL = LOCAL_OUTPUT_ROOT / "run_metadata.json"
