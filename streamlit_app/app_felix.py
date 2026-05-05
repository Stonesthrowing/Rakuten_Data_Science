from pathlib import Path
import json
import re
from html import escape

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Rakuten Multimodal Product Data Classification",
    layout="wide",
)

APP_DIR  = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"   # shared data folder agreed upon by the team

MM_LATE_DIR  = DATA_DIR / "Streamlit" / "MM_CamemBERT_ConvNeXtBase_LateFusion"
MM_INTER_DIR = DATA_DIR / "Streamlit" / "MM_CamemBERT_ConvNeXtBase_IntermediateFusion"

# -------------------------------------------------------------------
# Search paths for image-model artifacts.
#
# Primary:  <project_root>/data/Streamlit/I12_ConvNeXT  (team-agreed location)
# Kaggle:   /kaggle/input/streamlit/...
#
# APP_DIR is kept as a final fallback so the app also works when all
# files are placed next to app.py.
# -------------------------------------------------------------------
LOCAL_CANDIDATES = [
    DATA_DIR / "Streamlit" / "I12_ConvNeXT",   # primary — team-agreed project data folder
    DATA_DIR / "Streamlit",
    DATA_DIR,
]

KAGGLE_DATASETS = [
    # New dedicated Kaggle dataset: https://www.kaggle.com/datasets/arturillenseer/streamlit
    Path("/kaggle/input/streamlit"),
    # Older combined dataset path kept as fallback.
    Path("/kaggle/input/rakuten-product-images-ml"),
]
KAGGLE_CANDIDATES = []
for KAGGLE_DATASET in KAGGLE_DATASETS:
    KAGGLE_CANDIDATES.extend([
        KAGGLE_DATASET / "streamlit" / "I12_ConvNeXT",
        KAGGLE_DATASET / "streamlit",
        KAGGLE_DATASET / "Streamlit" / "I12_ConvNeXT",
        KAGGLE_DATASET / "Streamlit",
        # If the dataset root itself is the Streamlit folder.
        KAGGLE_DATASET / "I12_ConvNeXT",
        KAGGLE_DATASET,
    ])

# Fallback: also scan every Kaggle input dataset in case Kaggle changes the
# mounted folder name or the app is copied to a different dataset later.
KAGGLE_INPUT = Path("/kaggle/input")
if KAGGLE_INPUT.exists():
    try:
        for child in KAGGLE_INPUT.iterdir():
            if child.is_dir():
                KAGGLE_CANDIDATES.extend([
                    child / "streamlit" / "I12_ConvNeXT",
                    child / "streamlit",
                    child / "Streamlit" / "I12_ConvNeXT",
                    child / "Streamlit",
                    child,
                ])
    except Exception:
        pass

APP_FALLBACK_CANDIDATES = [
    APP_DIR,
]

SEARCH_DIRS = []
for base in LOCAL_CANDIDATES + KAGGLE_CANDIDATES + APP_FALLBACK_CANDIDATES:
    SEARCH_DIRS.extend([
        base,
        base / "artifacts",
        base / "artifacts" / "image_model",
        base / "artifacts" / "convnext_model",
        base / "artifacts" / "gradcam",
        base / "gradcam",
        base / "outputs",
        base / "outputs" / "convnext_base_final_v1",
        base / "convnext_base_final_v1",
        base / "I12_ConvNeXT",
    ])

_seen = set()
SEARCH_DIRS = [p for p in SEARCH_DIRS if not (str(p) in _seen or _seen.add(str(p)))]

def find_file(names):
    if isinstance(names, str):
        names = [names]
    for d in SEARCH_DIRS:
        for name in names:
            p = d / name
            if p.exists():
                return p
    return None


def find_gradcam_images():
    patterns = ["*gradcam*.png", "*GradCAM*.png", "*gradcam*.jpg", "*GradCAM*.jpg"]
    files = []
    for d in SEARCH_DIRS:
        if d.exists() and d.is_dir():
            for pattern in patterns:
                files.extend(d.glob(pattern))
                files.extend((d / "overlays").glob(pattern) if (d / "overlays").exists() else [])
    # Deduplicate while preserving filename-sorted presentation order.
    unique = {}
    for f in files:
        unique[str(f.resolve())] = f
    return sorted(unique.values(), key=lambda p: p.name)


FILES = {
    "metadata": find_file(["model_metadata.json", "convnext_model_metadata.json", "model_metadata_convnext_i12.json"]),
    "checkpoint": find_file(["best_model_base.pt", "best_model.pt"]),
    "history": find_file(["history.csv", "training_log.csv"]),
    "classification_report": find_file(["val_classification_report.txt", "classification_report.txt"]),
    "confusion_png": find_file(["val_confusion_matrix.png", "confusion_matrix.png"]),
    "confusion_npy": find_file(["confusion_matrix.npy", "val_confusion_matrix.npy"]),
    "learning_curves_png": find_file(["learning_curves.png", "training_history.png"]),
    "predictions": find_file(["val_predictions.csv", "predictions.csv"]),
    "id2label": find_file(["id2label.json", "index_to_label.json"]),
    "label2id": find_file(["label2id.json", "label_mapping.json"]),
    "logits": find_file(["val_logits_base.npy", "val_logits.npy", "image_val_logits.npy"]),
    "gradcam_predictions": find_file(["gradcam_prediction_table.csv"]),
    "gradcam_selected": find_file(["selected_gradcam_examples_4_groups.csv", "gradcam_index.csv"]),
}

DEFAULT_METADATA = {
    "display_name": "ConvNeXT-Base Image Model with Augmentation, Fully Unfrozen",
    "framework": "PyTorch",
    "architecture": "ConvNeXT-Base",
    "pretrained": True,
    "augmentation": "With augmentation / true",
    "freezing": "Fully unfrozen / fine-tuned",
    "checkpoint_file": "best_model_base.pt",
    "image_size": 224,
    "run_name": "convnext_base_final_v1",
    "max_epochs": 20,
    "batch_size": 16,
}

DEFAULT_ID2LABEL = {
    "0": 10, "1": 40, "2": 50, "3": 60, "4": 1140, "5": 1160, "6": 1180,
    "7": 1280, "8": 1281, "9": 1300, "10": 1301, "11": 1302, "12": 1320,
    "13": 1560, "14": 1920, "15": 1940, "16": 2060, "17": 2220, "18": 2280,
    "19": 2403, "20": 2462, "21": 2522, "22": 2582, "23": 2583, "24": 2585,
    "25": 2705, "26": 2905,
}

CATEGORY_NAMES = {
    "10": "Books",
    "40": "PC and console video games",
    "50": "Video game accessories",
    "60": "Video game consoles",
    "1140": "Geek merchandise and figurines",
    "1160": "Collectible cards",
    "1180": "Collectible board-game figurines",
    "1280": "Toys, plush toys and dolls",
    "1281": "Board and card games",
    "1300": "Toy cars and models",
    "1301": "Baby/child accessories and game furniture",
    "1302": "Outdoor games",
    "1320": "Women’s bags and early-childhood accessories",
    "1560": "Home furniture, decoration and storage",
    "1920": "Household linen",
    "1940": "Food and groceries",
    "2060": "Home lamps and decorative accessories",
    "2220": "Pet accessories",
    "2280": "Magazines",
    "2403": "Books and comics",
    "2462": "Video-game consoles and games",
    "2522": "Stationery and office storage",
    "2582": "Outdoor furniture and accessories",
    "2583": "Swimming-pool accessories",
    "2585": "Tools and gardening accessories",
    "2705": "Comics and books",
    "2905": "Downloadable games",
}

TABLE_HEIGHT = 680
BIG_TABLE_HEIGHT = 760
GRAPH_WIDTH = None
GRADCAM_WIDTH = 1050

# Fixed presentation order for interpretability examples.
# The dashboard shows up to three examples per category in this order.
GRADCAM_GROUP_ORDER = [
    "high_confidence_right_prediction",
    "high_confidence_wrong_prediction",
    "low_confidence_right_prediction",
    "low_confidence_wrong_prediction",
]
GRADCAM_EXAMPLES_PER_GROUP = 3

# One scenic city/church example was visually misleading for the presentation.
# Excluding it here makes Streamlit automatically use the next available example
# from the same interpretability category.
EXCLUDED_GRADCAM_LABEL_PAIRS = {("2060", "2522")}

def fit_table_height(df, max_height=TABLE_HEIGHT, min_height=78, row_height=36):
    """Height for scrollable dataframes without visible empty rows for short tables."""
    try:
        n_rows = len(df)
    except Exception:
        return max_height
    if n_rows <= 0:
        return min_height
    return min(max_height, max(min_height, (n_rows + 1) * row_height + 8))


def _format_cell(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    return escape(str(value))


def render_html_table(df, max_width="100%", compact=False):
    """Readable, wrapping HTML table for short report tables. No virtual empty rows."""
    if df is None or df.empty:
        st.info("No table data available.")
        return
    font = "1.02rem" if compact else "1.06rem"
    pad = "0.45rem 0.6rem" if compact else "0.62rem 0.75rem"
    header = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_format_cell(row[c])}</td>" for c in df.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    html = f'''
    <div class="table-scroll" style="max-width:{max_width}; overflow-x:auto; margin:0.35rem 0 1.0rem 0;">
      <table class="report-table" style="width:100%; border-collapse:collapse;">
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    <style>
    .report-table th {{
        background:#f7f8fa; color:#4b5563; font-weight:600; text-align:left;
        border:1px solid #e5e7eb; padding:{pad}; font-size:{font}; line-height:1.35;
        vertical-align:top; white-space:normal;
    }}
    .report-table td {{
        border:1px solid #e5e7eb; padding:{pad}; font-size:{font}; line-height:1.35;
        vertical-align:top; white-space:normal; color:#262730;
    }}
    </style>
    '''
    st.markdown(html, unsafe_allow_html=True)


def category_display(label):
    if pd.isna(label):
        return "—"
    key = str(int(label)) if isinstance(label, float) and label.is_integer() else str(label)
    name = CATEGORY_NAMES.get(key)
    return f"{key} — {name}" if name else key


def render_best_model_card(row):
    """Large, readable card for the best image-only model instead of a small one-row dataframe."""
    def val(name):
        v = row.get(name, "—")
        if pd.isna(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}" if name == "Macro F1" else f"{v:.3f}"
        return str(v)
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb; border-radius:0.9rem; padding:1.1rem 1.25rem; background:#fafafa; margin:0.6rem 0 1rem 0;">
          <div style="font-size:1.45rem; font-weight:800; line-height:1.25; color:#262730; margin-bottom:0.25rem;">{val('Model')}</div>
          <div style="font-size:1.05rem; color:#4b5563; margin-bottom:0.9rem;">
            {val('Family')} · {val('Training strategy')} · {val('Augmentation')} augmentation · {val('Image size')} px
          </div>
          <div style="display:grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap:0.75rem;">
            <div><div style="font-size:0.82rem; color:#6b7280;">Accuracy</div><div style="font-size:1.55rem; font-weight:700;">{val('Accuracy')}</div></div>
            <div><div style="font-size:0.82rem; color:#6b7280;">Macro F1</div><div style="font-size:1.55rem; font-weight:700;">{val('Macro F1')}</div></div>
            <div><div style="font-size:0.82rem; color:#6b7280;">Weighted F1</div><div style="font-size:1.55rem; font-weight:700;">{val('Weighted F1')}</div></div>
            <div><div style="font-size:0.82rem; color:#6b7280;">Best epoch</div><div style="font-size:1.55rem; font-weight:700;">{val('Best epoch')}</div></div>
            <div><div style="font-size:0.82rem; color:#6b7280;">Hardware / time</div><div style="font-size:1.55rem; font-weight:700; line-height:1.15;">{val('Hardware / time')}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# Static report text for the image-modeling chapter.
# -------------------------------------------------------------------
IMAGE_MODEL_RESULTS = [
    {"Model": "Model_I1_CNN128_NoAug_FromScratch", "Family": "CNN baseline", "Image size": "128", "Training strategy": "From scratch", "Augmentation": "No", "Accuracy": 0.5746, "Macro F1": 0.5065, "Weighted F1": 0.5643, "Best epoch": "30", "Hardware / time": "Tesla T4 / 203 min"},
    {"Model": "Model_I2_CNN128_ModerateAug_FromScratch", "Family": "CNN baseline", "Image size": "128", "Training strategy": "From scratch", "Augmentation": "Moderate", "Accuracy": 0.5694, "Macro F1": 0.4984, "Weighted F1": 0.5578, "Best epoch": "45", "Hardware / time": "Tesla T4 / not stated"},
    {"Model": "Model_I3_CNN256_NoAug_FromScratch", "Family": "CNN baseline", "Image size": "256", "Training strategy": "From scratch", "Augmentation": "No", "Accuracy": 0.5767, "Macro F1": 0.5090, "Weighted F1": 0.5627, "Best epoch": "33", "Hardware / time": "Tesla T4 / 68.76 min"},
    {"Model": "Model_I5_ResNet50_NoAug_Frozen", "Family": "ResNet", "Image size": "224", "Training strategy": "Frozen pretrained backbone", "Augmentation": "No", "Accuracy": 0.5948, "Macro F1": 0.5540, "Weighted F1": 0.5888, "Best epoch": "15", "Hardware / time": "123.9 min; 2x Tesla T4 / Kaggle"},
    {"Model": "Model_I6_ResNet50_ModerateAug_Partial", "Family": "ResNet", "Image size": "224", "Training strategy": "Partial unfreezing", "Augmentation": "Moderate", "Accuracy": 0.6752, "Macro F1": 0.6410, "Weighted F1": 0.6718, "Best epoch": "17", "Hardware / time": "58.44 min; RTX 5070 Ti"},
    {"Model": "Model_I7_ResNet50_ModerateAug_Full", "Family": "ResNet", "Image size": "224", "Training strategy": "Full unfreezing", "Augmentation": "Moderate", "Accuracy": 0.6849, "Macro F1": 0.6533, "Weighted F1": 0.6842, "Best epoch": "10", "Hardware / time": "111.65 min; RTX 5070 Ti"},
    {"Model": "Model_I8_ResNet50_ModerateAug_FromScratch", "Family": "ResNet", "Image size": "224", "Training strategy": "Random initialization", "Augmentation": "Moderate", "Accuracy": 0.5768, "Macro F1": 0.5105, "Weighted F1": 0.5601, "Best epoch": "18", "Hardware / time": "329.37 min; hardware not explicitly stated"},
    {"Model": "Model_I8b_ResNet101_NoAug_Frozen", "Family": "ResNet", "Image size": "224", "Training strategy": "Frozen pretrained backbone", "Augmentation": "No", "Accuracy": "0.5425-0.5701", "Macro F1": "0.4971-0.5212", "Weighted F1": "—", "Best epoch": "16", "Hardware / time": "hardware/time incomplete"},
    {"Model": "Model_I9_ConvNeXt_Tiny_ModerateAug_Full", "Family": "ConvNeXt", "Image size": "224", "Training strategy": "Full unfreeze", "Augmentation": "Moderate", "Accuracy": 0.7144, "Macro F1": 0.6850, "Weighted F1": 0.7112, "Best epoch": "19", "Hardware / time": "71.04 min; RTX 5070 Ti"},
    {"Model": "Model_I10_EfficientNetB0_NoAug_Partial", "Family": "EfficientNet", "Image size": "224", "Training strategy": "Partial fine-tuning", "Augmentation": "No", "Accuracy": 0.6173, "Macro F1": 0.5684, "Weighted F1": 0.6089, "Best epoch": "-", "Hardware / time": "33.18 min; RTX PRO 6000 Blackwell SE"},
    {"Model": "Model_I11_EfficientNetB0_ModerateAug_Partial", "Family": "EfficientNet", "Image size": "224", "Training strategy": "Partial fine-tuning", "Augmentation": "Moderate", "Accuracy": 0.5990, "Macro F1": 0.5489, "Weighted F1": 0.5892, "Best epoch": "-", "Hardware / time": "CUDA GPU; duration not fixed"},
    {"Model": "Model_I12_ConvNeXt_Base_ModerateAug_Full", "Family": "ConvNeXt", "Image size": "224", "Training strategy": "Full unfreeze", "Augmentation": "Moderate", "Accuracy": 0.7200, "Macro F1": 0.6924, "Weighted F1": 0.7200, "Best epoch": "20", "Hardware / time": "160.82 min; RTX 5070 Ti"},
    {"Model": "Model_I13_DINOv2_TrainAug_Frozen", "Family": "DINOv2", "Image size": "224", "Training strategy": "Frozen pretrained backbone", "Augmentation": "Training transform from timm", "Accuracy": 0.6647, "Macro F1": 0.6199, "Weighted F1": None, "Best epoch": "-", "Hardware / time": "~60-62 min/epoch; Apple Silicon MPS"},
]

# -------------------------------------------------------------------

AUGMENTATION_GROUPS = [
    {
        "Group": "No augmentation",
        "Models": "I1, I3, I5, I8b, I10",
        "Resize / crop": "Resize to model input size; no random crop.",
        "Geometric changes": "None.",
        "Color / policy changes": "None.",
        "Tensor + normalization": "ToTensor + ImageNet normalization where pretrained backbones are used.",
        "Comment": "Stable baseline because product images are usually upright and photographed in fairly consistent catalog-like conditions."
    },
    {
        "Group": "Moderate augmentation — ConvNeXt exact implementation",
        "Models": "I9, I12",
        "Resize / crop": "RandomResizedCrop(224, scale=(0.7, 1.0)).",
        "Geometric changes": "RandomHorizontalFlip().",
        "Color / policy changes": "TrivialAugmentWide().",
        "Tensor + normalization": "ToTensor + Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).",
        "Comment": "Identical transform confirmed in both the ConvNeXt-Tiny (I9) and ConvNeXt-Base (I12) training scripts."
    },
    {
        "Group": "Moderate augmentation — all other models",
        "Models": "I2, I6, I7, I8, I11",
        "Resize / crop": "Moderate random cropping / resizing reported where applicable; exact settings vary or are not present in the current app artifact set.",
        "Geometric changes": "Moderate flips, rotations, or translations depending on the specific notebook.",
        "Color / policy changes": "Moderate color, contrast, or policy augmentation depending on the specific notebook.",
        "Tensor + normalization": "Model-specific preprocessing and normalization.",
        "Comment": "Grouped together to avoid falsely claiming that every moderate run used the exact ConvNeXt transform."
    },
    {
        "Group": "Training transform from timm",
        "Models": "I13",
        "Resize / crop": "timm/DINOv2 training transform.",
        "Geometric changes": "Defined by the timm configuration used in the original DINOv2 notebook.",
        "Color / policy changes": "Defined by the timm configuration used in the original DINOv2 notebook.",
        "Tensor + normalization": "timm/DINOv2 preprocessing.",
        "Comment": "Reported separately because it is not the same augmentation family as the CNN/ResNet/ConvNeXt experiments."
    },
]

AUGMENTATION_DECISION_ROWS = [
    {"Point": "Why the report says moderate augmentation", "Explanation": "A stronger augmentation setup was tested, but it degraded performance substantially. The report notes validation accuracy around 0.21, training accuracy around 0.17, and early stopping after only three epochs."},
    {"Point": "Why stronger augmentation likely hurt", "Explanation": "The dataset already contains high visual variety across many product categories. Excessive synthetic variation can make visually distinct product cues less stable and increase class confusion."},
    {"Point": "Why product images need careful augmentation", "Explanation": "Many product photos are upright, centered, and photographed under similar catalog-like conditions. Large rotations, translations, or zooms can create unrealistic examples rather than better generalization."},
    {"Point": "Practical conclusion", "Explanation": "Moderate augmentation is a compromise: it adds robustness without destroying the original product structure that the image model needs for classification."},
]
# Loaders and helpers
# -------------------------------------------------------------------
@st.cache_data
def load_json(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_text(path):
    if path is None:
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data
def load_csv(path):
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_npy(path):
    if path is None:
        return None
    return np.load(path)


def parse_report_metrics(report_text):
    metrics = {}
    acc = re.search(r"\n\s*accuracy\s+([0-9.]+)\s+(\d+)", "\n" + report_text)
    if acc:
        metrics["accuracy"] = float(acc.group(1))
        metrics["validation_samples"] = int(acc.group(2))
    macro = re.search(r"\n\s*macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", "\n" + report_text)
    if macro:
        metrics["macro_precision"] = float(macro.group(1))
        metrics["macro_recall"] = float(macro.group(2))
        metrics["macro_f1"] = float(macro.group(3))
    weighted = re.search(r"\n\s*weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", "\n" + report_text)
    if weighted:
        metrics["weighted_precision"] = float(weighted.group(1))
        metrics["weighted_recall"] = float(weighted.group(2))
        metrics["weighted_f1"] = float(weighted.group(3))
    return metrics


def prepare_predictions(df):
    if df.empty:
        return df
    out = df.copy()
    if "true_label" not in out.columns and "prdtypecode" in out.columns:
        out["true_label"] = out["prdtypecode"]
    if "pred_label" not in out.columns and "pred" in out.columns:
        out["pred_label"] = out["pred"]
    if "correct" not in out.columns and {"true_label", "pred_label"}.issubset(out.columns):
        out["correct"] = out["true_label"].astype(str) == out["pred_label"].astype(str)
    if "image_filename" not in out.columns and {"productid", "imageid"}.issubset(out.columns):
        out["image_filename"] = out.apply(lambda r: f"image_{r['imageid']}_product_{r['productid']}.jpg", axis=1)
    return out


def format_path(p):
    return str(p) if p else "Not found"


def find_image_path(row, image_dirs):
    # Prefer the original absolute path when it exists.
    if "image_path_local" in row and isinstance(row.get("image_path_local"), str):
        p = Path(row.get("image_path_local"))
        if p.exists():
            return p
    filename = row.get("image_filename")
    if not filename:
        return None
    for d in image_dirs:
        p = Path(d.strip()) / filename
        if p.exists():
            return p
    return None


def format_value(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def placeholder_page(title):
    st.title("Rakuten Multimodal Product Data Classification")
    st.header(title)
    st.info("This chapter is intentionally left blank in this presentation version.")


def summary_table(metadata, metrics, id2label, preds):
    rows = [
        {"Section": "Model", "Field": "Model / chapter", "Value": "5.2 Best model: ConvNeXT-Base"},
        {"Section": "Model", "Field": "Display name", "Value": metadata.get("display_name")},
        {"Section": "Model", "Field": "Architecture", "Value": metadata.get("architecture")},
        {"Section": "Model", "Field": "Framework", "Value": metadata.get("framework")},
        {"Section": "Model", "Field": "Pretrained", "Value": format_value(metadata.get("pretrained"))},
        {"Section": "Model", "Field": "Augmentation", "Value": metadata.get("augmentation")},
        {"Section": "Model", "Field": "Freezing", "Value": metadata.get("freezing")},
        {"Section": "Model", "Field": "Image size", "Value": metadata.get("image_size")},
        {"Section": "Model", "Field": "Checkpoint", "Value": "Available" if FILES["checkpoint"] else "Not found"},
        {"Section": "Training", "Field": "Max epochs", "Value": metadata.get("max_epochs")},
        {"Section": "Training", "Field": "Batch size", "Value": metadata.get("batch_size")},
        {"Section": "Validation", "Field": "Classes", "Value": len(id2label) or 27},
        {"Section": "Validation", "Field": "Validation samples", "Value": metrics.get("validation_samples", len(preds) if not preds.empty else None)},
        {"Section": "Validation", "Field": "Accuracy", "Value": metrics.get("accuracy")},
        {"Section": "Validation", "Field": "Macro F1", "Value": metrics.get("macro_f1")},
        {"Section": "Validation", "Field": "Weighted F1", "Value": metrics.get("weighted_f1")},
        {"Section": "Validation", "Field": "Macro precision", "Value": metrics.get("macro_precision")},
        {"Section": "Validation", "Field": "Macro recall", "Value": metrics.get("macro_recall")},
    ]
    df = pd.DataFrame(rows)
    df["Value"] = df["Value"].apply(format_value)
    return df



def parse_classification_report_table(report_text, id2label):
    """Convert sklearn text classification_report into presentation-friendly tables."""
    class_rows = []
    summary_rows = []
    if not report_text:
        return pd.DataFrame(), pd.DataFrame()

    for line in report_text.splitlines():
        m = re.match(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", line)
        if m:
            idx = m.group(1)
            product_class = id2label.get(idx, id2label.get(str(idx), idx)) if id2label else idx
            class_rows.append({
                "Class index": int(idx),
                "Product class": product_class,
                "Precision": float(m.group(2)),
                "Recall": float(m.group(3)),
                "F1-score": float(m.group(4)),
                "Support": int(m.group(5)),
            })
            continue

        m = re.match(r"^\s*accuracy\s+([0-9.]+)\s+(\d+)\s*$", line)
        if m:
            summary_rows.append({
                "Metric": "Accuracy",
                "Precision": "—",
                "Recall": "—",
                "F1-score / score": float(m.group(1)),
                "Support": int(m.group(2)),
            })
            continue

        m = re.match(r"^\s*(macro avg|weighted avg)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", line)
        if m:
            summary_rows.append({
                "Metric": m.group(1).title(),
                "Precision": float(m.group(2)),
                "Recall": float(m.group(3)),
                "F1-score / score": float(m.group(4)),
                "Support": int(m.group(5)),
            })

    return pd.DataFrame(class_rows), pd.DataFrame(summary_rows)


def render_metric_cards(metrics):
    acc = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    weighted_f1 = metrics.get("weighted_f1")
    support = metrics.get("validation_samples")
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, int):
            return f"{v:,}"
        return f"{v:.3f}"
    st.markdown(
        f"""
        <style>
        .metric-grid {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin: 1rem 0 1.5rem 0;}}
        .metric-card {{border: 1px solid #e5e7eb; border-radius: 0.75rem; padding: 1rem 1.2rem; background: #fafafa;}}
        .metric-card-label {{font-size: 0.95rem; color: #4b5563; margin-bottom: 0.35rem;}}
        .metric-card-value {{font-size: 2.0rem; font-weight: 700; color: #262730; line-height: 1.1;}}
        </style>
        <div class="metric-grid">
          <div class="metric-card"><div class="metric-card-label">Accuracy</div><div class="metric-card-value">{fmt(acc)}</div></div>
          <div class="metric-card"><div class="metric-card-label">Macro F1</div><div class="metric-card-value">{fmt(macro_f1)}</div></div>
          <div class="metric-card"><div class="metric-card-label">Weighted F1</div><div class="metric-card-value">{fmt(weighted_f1)}</div></div>
          <div class="metric-card"><div class="metric-card-label">Validation samples</div><div class="metric-card-value">{fmt(support)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def readable_gradcam_group(group):
    text = str(group or "Grad-CAM example")
    confidence = "↓ Low confidence" if "low_confidence" in text else "↑ High confidence" if "high_confidence" in text else "Confidence"
    correctness = "Wrong prediction" if "wrong_prediction" in text else "Right prediction" if "right_prediction" in text else "Prediction"
    color = "#b91c1c" if "wrong_prediction" in text else "#15803d" if "right_prediction" in text else "#262730"
    return confidence, correctness, color


def infer_gradcam_group(confidence, correct):
    try:
        conf = float(confidence)
    except Exception:
        conf = None
    confidence_part = "low_confidence" if conf is not None and conf < 0.5 else "high_confidence"
    correct_part = "right_prediction" if bool(correct) else "wrong_prediction"
    return f"{confidence_part}_{correct_part}"


def gradcam_observation(row):
    group = str(row.get("group", ""))
    true_label = str(row.get("true_label", ""))
    pred_label = str(row.get("pred_label", ""))
    specific = {
        ("1280", "2583"): "Focus appears diffuse and partly off-object/background; this fits a low-confidence wrong prediction.",
        ("1302", "1560"): "Focus is on a visible object detail, but the model appears to confuse the product type.",
        ("2060", "2522"): "Focus is near the image edge rather than the central product, suggesting background or framing influence.",
        ("1280", "2522"): "Focus is on a subset of the items rather than the whole product group; this may explain the stationery-like prediction.",
        ("1560", "2522"): "Focus is on the label/front panel, which may make the item look like stationery rather than home furniture/decor.",
    }
    if (true_label, pred_label) in specific:
        return specific[(true_label, pred_label)]
    if "right_prediction" in group and "high_confidence" in group:
        return "Sensible focus on the visible product region; the heatmap supports the correct high-confidence prediction."
    if "right_prediction" in group and "low_confidence" in group:
        return "Correct but uncertain; the focus is less decisive or the visual evidence is ambiguous."
    if "wrong_prediction" in group and "high_confidence" in group:
        return "High-confidence mistake; inspect whether the model focuses on background, packaging text, or a misleading product detail."
    if "wrong_prediction" in group and "low_confidence" in group:
        return "Low-confidence error; the image likely contains ambiguous cues or diffuse attention."
    return "Inspect whether the highlighted region corresponds to a plausible product cue."


def render_gradcam_header(row):
    confidence_label, correctness_label, color = readable_gradcam_group(row.get("group", ""))
    true_label = category_display(row.get("true_label", "—"))
    pred_label = category_display(row.get("pred_label", "—"))
    conf = row.get("confidence", None)
    conf_text = f"{float(conf):.3f}" if pd.notna(conf) else "—"
    cam_target = category_display(row.get("cam_target", row.get("pred_label", "—")))
    st.markdown(
        f"""
        <div style="margin-top: 1.4rem; margin-bottom: 0.55rem;">
          <div style="font-size: 1.35rem; font-weight: 700; line-height: 1.2;">
            <span style="color: #262730;">{confidence_label}</span>
            <span style="color: #6b7280;"> — </span>
            <span style="color: {color};">{correctness_label}</span>
          </div>
          <div style="font-size: 1.02rem; color: #262730; margin-top: 0.25rem;">
            Correct category: <b>{true_label}</b><br/>
            Predicted category: <b>{pred_label}</b><br/>
            Confidence: <b>{conf_text}</b> &nbsp;|&nbsp; CAM target: <b>{cam_target}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_observation_box(row):
    st.markdown(
        f"""
        <div style="border: 1px solid #e5e7eb; border-radius: 0.7rem; padding: 0.9rem 1rem; background: #fafafa; margin-top: 1.0rem;">
          <div style="font-weight: 700; margin-bottom: 0.35rem;">Observation</div>
          <div style="font-size: 0.98rem; color: #374151; line-height: 1.45;">{gradcam_observation(row)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def load_gradcam_display_image(path):
    """Load a pre-rendered Grad-CAM panel and remove the old text title embedded above the pictures."""
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        # The saved Grad-CAM panels contain a result string at the top.
        # Cropping about 8% removes that result line while keeping the actual panels.
        crop_top = min(max(int(h * 0.135), 95), 150)
        return img.crop((0, crop_top, w, h))
    except Exception:
        return str(path)


def parse_gradcam_filename(path, id2label):
    name = path.name
    # Example: 01_gradcam_idx2101_true11_pred11_conf0.999_pred.png
    m = re.search(r"idx(?P<idx>\d+)_true(?P<true_id>\d+)_pred(?P<pred_id>\d+)_conf(?P<conf>[0-9.]+)_", name)
    if not m:
        return {"file": path, "filename": name, "group": "Grad-CAM example"}
    label_lookup = {**DEFAULT_ID2LABEL, **{str(k): v for k, v in (id2label or {}).items()}}
    true_id = m.group("true_id")
    pred_id = m.group("pred_id")
    true_label = label_lookup.get(str(true_id), true_id)
    pred_label = label_lookup.get(str(pred_id), pred_id)
    conf = float(m.group("conf"))
    correct = str(true_label) == str(pred_label)
    return {
        "file": path,
        "filename": name,
        "idx": int(m.group("idx")),
        "true_id": int(true_id),
        "pred_id": int(pred_id),
        "true_label": true_label,
        "pred_label": pred_label,
        "confidence": conf,
        "correct": correct,
        "group": infer_gradcam_group(conf, correct),
    }


def prepare_gradcam_table(images, selected_df, id2label):
    rows = [parse_gradcam_filename(p, id2label) for p in images]
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Enrich group from selected_gradcam_examples_4_groups.csv when possible.
    if not selected_df.empty and {"true_label", "pred_label", "confidence", "group"}.issubset(selected_df.columns):
        groups = []
        for _, r in out.iterrows():
            candidates = selected_df.copy()
            candidates = candidates[candidates["true_label"].astype(str) == str(r.get("true_label"))]
            candidates = candidates[candidates["pred_label"].astype(str) == str(r.get("pred_label"))]
            if "confidence" in candidates.columns and not candidates.empty:
                candidates = candidates.assign(_dist=(candidates["confidence"].astype(float) - float(r.get("confidence", 0))).abs())
                candidates = candidates.sort_values("_dist")
                if not candidates.empty and candidates.iloc[0]["_dist"] < 0.002:
                    groups.append(candidates.iloc[0]["group"])
                    continue
            groups.append(r.get("group", infer_gradcam_group(r.get("confidence"), r.get("correct"))))
        out["group"] = groups
    # Ensure every row has a meaningful group even when no selected CSV was available.
    if "group" not in out.columns or out["group"].eq("Grad-CAM example").any():
        out["group"] = out.apply(lambda r: infer_gradcam_group(r.get("confidence"), r.get("correct")), axis=1)
    return out


def make_label_table(id2label):
    if not id2label:
        return pd.DataFrame()
    label_rows = [{"Index": int(k), "Product class": v} for k, v in id2label.items()]
    labels_df = pd.DataFrame(label_rows).sort_values("Index")
    labels_df["Product class"] = labels_df["Product class"].apply(category_display)
    return labels_df


def is_excluded_gradcam_example(row):
    true_label = str(row.get("true_label", ""))
    pred_label = str(row.get("pred_label", ""))
    return (true_label, pred_label) in EXCLUDED_GRADCAM_LABEL_PAIRS


def ordered_gradcam_examples(gradcam_df):
    if gradcam_df.empty:
        return gradcam_df
    out = gradcam_df.copy()
    out = out[~out.apply(is_excluded_gradcam_example, axis=1)]
    out["_group_order"] = out["group"].apply(lambda g: GRADCAM_GROUP_ORDER.index(g) if g in GRADCAM_GROUP_ORDER else len(GRADCAM_GROUP_ORDER))
    # Within each group, keep examples deterministic and easy to review.
    if "confidence" in out.columns:
        out = out.sort_values(["_group_order", "confidence", "filename"], ascending=[True, False, True])
    else:
        out = out.sort_values(["_group_order", "filename"], ascending=[True, True])
    return out.drop(columns=["_group_order"], errors="ignore")


metadata = load_json(FILES["metadata"]) or DEFAULT_METADATA
report_text = load_text(FILES["classification_report"])
metrics = parse_report_metrics(report_text)
id2label = {**DEFAULT_ID2LABEL, **{str(k): v for k, v in (load_json(FILES["id2label"]) or {}).items()}}
label2id = load_json(FILES["label2id"]) or {}
preds = prepare_predictions(load_csv(FILES["predictions"]))
history = load_csv(FILES["history"])
gradcam_selected = load_csv(FILES["gradcam_selected"])
gradcam_prediction_table = load_csv(FILES["gradcam_predictions"])
gradcam_images = find_gradcam_images()
gradcam_table = prepare_gradcam_table(gradcam_images, gradcam_selected, id2label)


def _mm_path(base, filename):
    p = base / filename
    return p if p.exists() else None

mm_late_meta     = load_json(_mm_path(MM_LATE_DIR,  "run_metadata.json")) or {}
mm_late_report   = load_text(_mm_path(MM_LATE_DIR,  "fusion_classification_report.txt")) or ""
mm_late_cm_png   = _mm_path(MM_LATE_DIR,  "confusion_matrix.png")
mm_late_preds    = prepare_predictions(load_csv(_mm_path(MM_LATE_DIR, "val_predictions.csv")))
mm_inter_meta    = load_json(_mm_path(MM_INTER_DIR, "run_metadata.json")) or {}
mm_inter_history = pd.DataFrame(load_json(_mm_path(MM_INTER_DIR, "history.json")) or [])
mm_inter_cm_png  = _mm_path(MM_INTER_DIR, "confusion_matrix.png")


# Global readability tweaks for 1080p and 4K screens.
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 17px; }
    .stMarkdown p, .stMarkdown li { font-size: 1.04rem; line-height: 1.48; }
    h1 { line-height: 1.15 !important; }
    h2, h3 { line-height: 1.22 !important; }
    div[data-testid="stDataFrame"] { font-size: 1.02rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Sidebar navigation: button-style navigation with chapter hierarchy.
# -------------------------------------------------------------------
st.sidebar.title("Rakuten Multimodal Product Data Classification")
st.sidebar.markdown("### Navigation")

NAV_LEVELS = {
    "1. Overview": 0,
    "2. Data Exploration": 0,
    "2.1 Text": 1,
    "2.2 Image": 1,
    "3. Preprocessing": 0,
    "4. Text Modeling": 0,
    "4.1 Overview": 1,
    "4.2 Best model": 1,
    "5. Image Modeling": 0,
    "5.1 Overview": 1,
    "5.2 Best model: ConvNeXT-Base — Summary": 1,
    "5.2.2 Training history": 2,
    "5.2.3 Classification Report + Confusion Matrix": 2,
    "5.2.5 Error Analysis + Interpretability": 2,
    "5.2.6 Setup Check": 2,
    "6. Multimodal": 0,
    "6.1 Overview": 1,
    "6.1.1 Simple Fusion": 2,
    "6.2 Best model — Summary": 1,
    "6.2.2 Training history": 2,
    "6.2.3 Classification Report + Confusion Matrix": 2,
    "6.2.5 Error Analysis": 2,
    "7. Prediction Tool": 0,
}
NAV_DISPLAY = {
    "1. Overview": "Overview",
    "2. Data Exploration": "Data Exploration",
    "2.1 Text": "Text",
    "2.2 Image": "Image",
    "3. Preprocessing": "Preprocessing",
    "4. Text Modeling": "Text Modeling",
    "4.1 Overview": "Overview Text models",
    "4.2 Best model": "Best model",
    "5. Image Modeling": "Image Modeling",
    "5.1 Overview": "Overview Image models",
    "5.2 Best model: ConvNeXT-Base — Summary": "Best model: ConvNeXT-Base — Summary",
    "5.2.2 Training history": "Training history Image model",
    "5.2.3 Classification Report + Confusion Matrix": "Classification Report + Confusion Matrix Image model",
    "5.2.5 Error Analysis + Interpretability": "Error Analysis + Interpretability Image model",
    "5.2.6 Setup Check": "Setup Check Image model",
    "6. Multimodal": "Multimodal",
    "6.1 Overview": "Overview Multimodal models",
    "6.1.1 Simple Fusion": "Simple Fusion",
    "6.2 Best model — Summary": "Best model — Summary",
    "6.2.2 Training history": "Training history Multimodal model",
    "6.2.3 Classification Report + Confusion Matrix": "Classification Report + Confusion Matrix Multimodal model",
    "6.2.5 Error Analysis": "Error Analysis Multimodal model",
    "7. Prediction Tool": "Prediction Tool",
}
NAV_ITEMS = list(NAV_LEVELS.keys())
DEFAULT_PAGE = "5.2 Best model: ConvNeXT-Base — Summary"

# Query-parameter navigation makes the sidebar look and behave like regular buttons,
# while keeping the selected chapter stable after Streamlit reruns.
def _get_query_page():
    try:
        raw = st.query_params.get("page")
    except Exception:
        raw = None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    return raw if raw in NAV_ITEMS else None

page = _get_query_page() or st.session_state.get("page", DEFAULT_PAGE)
if page not in NAV_ITEMS:
    page = DEFAULT_PAGE
st.session_state["page"] = page

from urllib.parse import quote

# Button-style navigation. The HTML is intentionally left-aligned with no
# leading indentation so Streamlit/Markdown does not render it as a code block.
nav_css = """
<style>
section[data-testid="stSidebar"] {
    background: #ffffff !important;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.35rem !important;
    line-height: 1.2 !important;
    margin-bottom: 0.75rem !important;
}
.sidebar-nav {
    margin-top: 0.2rem;
    margin-bottom: 0.9rem;
}
.sidebar-nav a,
.sidebar-nav a:link,
.sidebar-nav a:visited,
.sidebar-nav a:hover,
.sidebar-nav a:active {
    text-decoration: none !important;
    color: inherit !important;
}
.sidebar-nav .nav-button,
.sidebar-nav .nav-button * {
    text-decoration: none !important;
}
.nav-button {
    display: block;
    width: 100%;
    box-sizing: border-box;
    border: 1px solid #e5e7eb;
    border-radius: 0.55rem;
    background: #f3f4f6;
    margin-left: 0 !important;
    margin-right: 0 !important;
    line-height: 1.22;
    white-space: normal;
    overflow-wrap: anywhere;
    text-align: left;
    transition: background 0.12s ease, border-color 0.12s ease, box-shadow 0.12s ease;
}
.nav-button:hover {
    background: #e5e7eb;
    border-color: #cbd5e1;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
.nav-button.active {
    border-color: #ff4b4b;
    background: #fff1f1;
    color: #111827;
    font-weight: 700;
}
.nav-level-0 {
    color: #111827;
    font-size: 1.12rem;
    font-weight: 700;
    padding: 0.56rem 0.70rem;
    min-height: 2.45rem;
    margin-top: 0.54rem;
    margin-bottom: 0.16rem;
}
.nav-level-1 {
    color: #374151;
    font-size: 0.98rem;
    font-weight: 600;
    padding: 0.44rem 0.70rem;
    min-height: 2.18rem;
    margin-top: 0.12rem;
    margin-bottom: 0.12rem;
}
.nav-level-2 {
    color: #6b7280;
    font-size: 0.89rem;
    font-weight: 500;
    padding: 0.34rem 0.70rem;
    min-height: 1.95rem;
    margin-top: 0.06rem;
    margin-bottom: 0.06rem;
}
.sidebar-nav a:first-of-type .nav-button {
    margin-top: 0 !important;
}
</style>
"""
nav_html = [nav_css, '<div class="sidebar-nav">']
for item in NAV_ITEMS:
    label = escape(NAV_DISPLAY.get(item, item))
    level = NAV_LEVELS[item]
    active = " active" if item == page else ""
    href = f"?page={quote(item)}"
    nav_html.append(f'<a href="{href}" target="_self"><div class="nav-button nav-level-{level}{active}">{label}</div></a>')
nav_html.append("</div>")
st.sidebar.markdown("\n".join(nav_html), unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Image folders for Prediction Explorer / Error Analysis")
default_image_dirs = "\n".join([
    # Primary — team-agreed project data folder
    str(DATA_DIR / "Streamlit" / "I12_ConvNeXT" / "images" / "image_train"),
    str(DATA_DIR / "Streamlit" / "I12_ConvNeXT" / "image_train"),
    str(DATA_DIR / "images" / "image_train"),
    str(DATA_DIR / "image_train"),

    # Kaggle Streamlit dataset layout
    "/kaggle/input/streamlit/Streamlit/I12_ConvNeXT/images/image_train",
    "/kaggle/input/streamlit/Streamlit/I12_ConvNeXT/image_train",
    "/kaggle/input/streamlit/Streamlit/images/image_train",
    "/kaggle/input/streamlit/Streamlit/image_train",
    "/kaggle/input/streamlit/I12_ConvNeXT/images/image_train",
    "/kaggle/input/streamlit/I12_ConvNeXT/image_train",
    "/kaggle/input/streamlit/images/image_train",
    "/kaggle/input/streamlit/image_train",

    # Separate Kaggle image dataset
    "/kaggle/input/rakuten-product-images-ml/image_train",
    "/kaggle/input/rakuten-product-images-ml/images/image_train",
    "/kaggle/input/rakuten-product-images-ml/image_test",
    "/kaggle/input/rakuten-product-images-ml/images/image_test",

    # Kaggle older/fallback layouts
    "/kaggle/input/rakuten-product-images-ml/streamlit/I12_ConvNeXT/images/image_train",
    "/kaggle/input/rakuten-product-images-ml/streamlit/I12_ConvNeXT/image_train",
    "/kaggle/input/rakuten-product-images-ml/Streamlit/I12_ConvNeXT/images/image_train",
    "/kaggle/input/rakuten-product-images-ml/Streamlit/I12_ConvNeXT/image_train",
    str(APP_DIR / "images" / "image_train"),
    str(APP_DIR / "image_train"),
    "/kaggle/input/i12/images/image_train",
    "/kaggle/input/i5/images/image_train",
])
image_dir_text = st.sidebar.text_area("One folder per line", value=default_image_dirs, height=90)
image_dirs = [x for x in image_dir_text.splitlines() if x.strip()]

# -------------------------------------------------------------------
# Blank report chapters
# -------------------------------------------------------------------
if page not in [
    "1. Overview",
    "5. Image Modeling",
    "5.1 Overview",
    "5.2 Best model: ConvNeXT-Base — Summary",
    "5.2.2 Training history",
    "5.2.3 Classification Report + Confusion Matrix",
    "5.2.5 Error Analysis + Interpretability",
    "5.2.6 Setup Check",
    "6. Multimodal",
    "6.1 Overview",
    "6.1.1 Simple Fusion",
    "6.2 Best model — Summary",
    "6.2.2 Training history",
    "6.2.3 Classification Report + Confusion Matrix",
    "6.2.5 Error Analysis",
]:
    placeholder_page(page)

elif page == "1. Overview":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("1. Overview")
    st.write(
        "This Streamlit report presents the Rakuten product classification project with a focus on image-modeling results, "
        "the selected ConvNeXT-Base image model, and Grad-CAM interpretability examples."
    )

    st.subheader("Class labels")
    labels_df = make_label_table(id2label)
    if labels_df.empty:
        st.warning("No class-label mapping found.")
    else:
        render_html_table(labels_df, max_width="760px", compact=True)

    st.subheader("Report structure")
    overview_rows = pd.DataFrame([
        {"Chapter": "2-3", "Content": "Data exploration and preprocessing placeholders for the full written report."},
        {"Chapter": "4", "Content": "Text-modeling placeholders for the full written report."},
        {"Chapter": "5", "Content": "Image-modeling comparison, selected ConvNeXT-Base model, training history, validation metrics, errors, and interpretability."},
        {"Chapter": "6", "Content": "Multimodal-modeling placeholders for the full written report."},
        {"Chapter": "7", "Content": "Prediction-tool placeholder."},
    ])
    render_html_table(overview_rows, max_width="900px")

elif page == "5. Image Modeling":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("5. Image Modeling")
    st.write(
        "The image-modeling stage was designed as a structured progression from simple convolutional baselines "
        "to transfer learning and then to stronger pretrained architectures. This made it possible to isolate the "
        "effects of augmentation, input resolution, pretrained representations, fine-tuning depth, and model capacity."
    )
    st.subheader("Modeling approach")
    approach_rows = pd.DataFrame([
        {"Step": "1", "Area": "CNN baselines from scratch", "Purpose": "Establish an image-only baseline without pretrained features."},
        {"Step": "2", "Area": "Resolution and augmentation checks", "Purpose": "Test whether larger images or moderate augmentation improve generalization."},
        {"Step": "3", "Area": "ResNet transfer learning", "Purpose": "Compare frozen, partial, full fine-tuning, and from-scratch ResNet variants."},
        {"Step": "4", "Area": "Additional pretrained architectures", "Purpose": "Evaluate EfficientNet, ConvNeXt, and DINOv2 as stronger image backbones."},
        {"Step": "5", "Area": "Best image branch selection", "Purpose": "Select the strongest image model for later multimodal fusion."},
    ])
    render_html_table(approach_rows, max_width="900px")

elif page == "5.1 Overview":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("5.1 Overview Image models")
    st.write(
        "This overview summarizes all image-only models tried before the detailed ConvNeXT-Base discussion in chapter 5.2. "
        "The results show a progression from scratch CNNs to pretrained and fine-tuned image backbones."
    )
    results_df = pd.DataFrame(IMAGE_MODEL_RESULTS)
    display_df = results_df.copy().replace({None: "—"})
    st.subheader("Image model comparison")
    render_html_table(display_df)
    st.subheader("Best image-only model")
    best_rows = results_df[results_df["Model"].astype(str).str.contains("I12_ConvNeXt_Base", na=False)]
    if not best_rows.empty:
        render_best_model_card(best_rows.iloc[0])
    st.markdown(
        """
        The CNN baselines learned useful visual features, but their validation metrics remained clearly below the stronger pretrained models. Frozen ResNet50 improved over the scratch baselines, and partial or full fine-tuning improved further. The strongest available image-only result is the ConvNeXT-Base model with moderate augmentation and full unfreezing, which reaches the highest reported macro F1 in the image-modeling experiments.
        """
    )

    st.subheader("Augmentation definitions used in image-modeling runs")
    st.write(
        "The report distinguishes between no augmentation, moderate augmentation, and the timm/DINOv2 training transform. "
        "The exact I12 ConvNeXT-Base training transform is available in the ConvNeXT training script. "
        "For the other moderate runs, the app groups them as 'all other models' unless their exact source transform is available."
    )
    aug_df = pd.DataFrame(AUGMENTATION_GROUPS)
    render_html_table(aug_df)

    st.subheader("Why moderate augmentation was used")
    st.write(
        "A stronger augmentation setup was considered, but it performed poorly. The working interpretation is that the product-image dataset already contains large natural variety, while many images are still catalog-like: upright, centered, and photographed in similar orientations. Too much artificial transformation can therefore add confusion instead of revealing stable visual patterns."
    )
    decision_df = pd.DataFrame(AUGMENTATION_DECISION_ROWS)
    render_html_table(decision_df)

    st.subheader("Key conclusions")
    st.markdown(
        """
        - Transfer learning clearly outperformed training deeper backbones from scratch.
        - Partial and full fine-tuning gave meaningful gains over frozen feature extraction when suitable hardware was available.
        - Augmentation was not universally beneficial; preserving original product structure often mattered more than synthetic variation.
        - ConvNeXt produced the strongest image-only results in the available experiments and is therefore the best candidate image branch for multimodal fusion.
        - Image-only performance remained below the strongest text-only results, suggesting that images are most valuable as complementary information in a fused model.
        """
    )

    with st.expander("Open issues to verify before final report"):
        st.markdown(
            """
            - One summary fragment reports slightly different Model_I3 metrics. The final report should resolve this against the definitive artifact set.
            """
        )
elif page == "5.2 Best model: ConvNeXT-Base — Summary":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("5.2 Best model: ConvNeXT-Base — Summary")
    st.markdown(
        "ConvNeXT-Base (full fine-tuning, TrivialAugmentWide) reached **72.0 % accuracy and 0.692 macro F1** on the 16,984-sample validation set across 27 product classes, "
        "training for the full 20 epochs without early stopping — suggesting the model was still benefiting from updates at the end of training. "
        "Class-level performance is highly uneven: collectible cards (F1 = 0.96) and swimming-pool accessories (F1 = 0.88) are near-perfect, "
        "while board & card games (F1 = 0.40) and toys & plush toys (F1 = 0.47) remain the hardest categories — "
        "a pattern confirmed by the confusion matrix (5.2.3) and Grad-CAM visualisations (5.2.5), which show the model attends to plausible product regions for correctly classified images."
    )
    st.write(
        "This chapter presents the saved validation results for the ConvNeXT-Base image model. "
        "The app does not retrain the model and does not load the PyTorch checkpoint unless it is later extended for live inference."
    )

    st.subheader("Model and validation summary")
    summary_df = summary_table(metadata, metrics, id2label, preds)
    render_html_table(summary_df, max_width="100%")

    st.info("The class-label table is shown in chapter 1. Overview to introduce the target classes before the model chapters.")

elif page == "5.2.2 Training history":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("5.2.2 Training history")
    if FILES["learning_curves_png"]:
        st.image(str(FILES["learning_curves_png"]), caption="Saved training history plot", use_container_width=True)
    if not history.empty:
        st.subheader("History table")
        st.dataframe(history, hide_index=True, use_container_width=True, height=fit_table_height(history))
        if "epoch" in history.columns:
            for cols, title in [(["t_loss", "v_loss"], "Loss by epoch"), (["t_f1", "v_f1"], "Macro F1 by epoch")]:
                existing = [c for c in cols if c in history.columns]
                if existing:
                    fig, ax = plt.subplots(figsize=(10.8, 4.4))
                    for col in existing:
                        ax.plot(history["epoch"], history[col], marker="o", label=col)
                    ax.set_xlabel("Epoch")
                    ax.set_title(title)
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)
    else:
        st.warning("No history.csv found.")

elif page == "5.2.3 Classification Report + Confusion Matrix":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("5.2.3 Classification Report + Confusion Matrix")
    if report_text:
        class_report_df, summary_report_df = parse_classification_report_table(report_text, id2label)
        st.subheader("Main validation metrics")
        render_metric_cards(metrics)
        if not class_report_df.empty:
            st.subheader("Per-class classification report")
            class_display = class_report_df.copy()
            for c in ["Precision", "Recall", "F1-score"]:
                class_display[c] = class_display[c].map(lambda x: f"{x:.2f}")
            render_html_table(class_display, max_width="1050px", compact=True)
        if not summary_report_df.empty:
            st.subheader("Summary rows")
            summary_display = summary_report_df.copy()
            for c in ["Precision", "Recall", "F1-score / score"]:
                summary_display[c] = summary_display[c].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            render_html_table(summary_display, max_width="900px")
    else:
        st.warning("No classification report found.")

    st.subheader("Confusion matrix")
    if FILES["confusion_png"]:
        st.image(str(FILES["confusion_png"]), caption="Saved normalized validation confusion matrix", use_container_width=True)
    elif FILES["confusion_npy"]:
        cm = load_npy(FILES["confusion_npy"])
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        im = ax.imshow(cm)
        fig.colorbar(im, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("No confusion matrix image or NumPy file found.")

elif page == "5.2.5 Error Analysis + Interpretability":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("5.2.5 Error Analysis")
    if preds.empty:
        st.warning("No val_predictions.csv found.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        total = len(preds)
        correct = int(preds["correct"].sum()) if "correct" in preds.columns else 0
        c1.metric("Validation rows", f"{total:,}")
        c2.metric("Correct", f"{correct:,}")
        c3.metric("Errors", f"{total - correct:,}")
        c4.metric("Mean confidence", f"{preds['confidence'].mean():.3f}" if "confidence" in preds.columns else "—")

        filtered = preds.copy()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            view = st.selectbox("Prediction status", ["All", "Only errors", "Only correct"])
            if "correct" in filtered.columns:
                if view == "Only errors":
                    filtered = filtered[~filtered["correct"]]
                elif view == "Only correct":
                    filtered = filtered[filtered["correct"]]
        with col_b:
            if "true_label" in filtered.columns:
                classes = ["All"] + sorted(filtered["true_label"].astype(str).unique().tolist(), key=lambda x: int(x) if x.isdigit() else x)
                true_choice = st.selectbox("True class", classes)
                if true_choice != "All":
                    filtered = filtered[filtered["true_label"].astype(str) == true_choice]
        with col_c:
            if "pred_label" in filtered.columns:
                classes = ["All"] + sorted(filtered["pred_label"].astype(str).unique().tolist(), key=lambda x: int(x) if x.isdigit() else x)
                pred_choice = st.selectbox("Predicted class", classes)
                if pred_choice != "All":
                    filtered = filtered[filtered["pred_label"].astype(str) == pred_choice]

        if "confidence" in filtered.columns and not filtered.empty:
            conf_range = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), 0.01)
            filtered = filtered[(filtered["confidence"] >= conf_range[0]) & (filtered["confidence"] <= conf_range[1])]

        sort_options = [c for c in ["confidence", "true_label", "pred_label", "productid", "imageid"] if c in filtered.columns]
        if sort_options:
            sort_col = st.selectbox("Sort by", sort_options)
            ascending = st.checkbox("Ascending", value=False)
            filtered = filtered.sort_values(sort_col, ascending=ascending)

        st.subheader("Filtered validation predictions")
        st.dataframe(filtered, hide_index=True, use_container_width=True, height=fit_table_height(filtered, max_height=BIG_TABLE_HEIGHT))
        st.download_button(
            "Download filtered rows as CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            "filtered_convnext_predictions.csv",
            "text/csv",
        )

    st.divider()
    st.header("Interpretability: Grad-CAM")
    st.write(
        "Grad-CAM highlights image regions that contributed strongly to the selected class prediction. "
        "It is useful for inspecting whether the model focuses on plausible product regions, but it does not prove semantic understanding."
    )

    if gradcam_table.empty:
        st.warning("No Grad-CAM images found. Place files such as '*gradcam*.png' in the app folder, 'gradcam', or 'artifacts/gradcam'.")
    else:
        ordered_gdf = ordered_gradcam_examples(gradcam_table)
        st.caption(
            f"Showing up to {GRADCAM_EXAMPLES_PER_GROUP} example(s) per interpretability category "
            "in the fixed presentation order."
        )

        for group in GRADCAM_GROUP_ORDER:
            group_df = ordered_gdf[ordered_gdf["group"] == group].head(GRADCAM_EXAMPLES_PER_GROUP)
            confidence_label, correctness_label, _ = readable_gradcam_group(group)
            st.subheader(f"{confidence_label} — {correctness_label}")
            if group_df.empty:
                st.info("No example found for this category.")
                continue
            for _, row in group_df.iterrows():
                render_gradcam_header(row)
                image_col, note_col = st.columns([3, 1])
                with image_col:
                    st.image(load_gradcam_display_image(row["file"]), width=GRADCAM_WIDTH)
                with note_col:
                    render_observation_box(row)

        extra_groups = [g for g in ordered_gdf["group"].dropna().unique().tolist() if g not in GRADCAM_GROUP_ORDER]
        if extra_groups:
            with st.expander("Other Grad-CAM examples"):
                for group in extra_groups:
                    group_df = ordered_gdf[ordered_gdf["group"] == group].head(GRADCAM_EXAMPLES_PER_GROUP)
                    confidence_label, correctness_label, _ = readable_gradcam_group(group)
                    st.subheader(f"{confidence_label} — {correctness_label}")
                    for _, row in group_df.iterrows():
                        render_gradcam_header(row)
                        image_col, note_col = st.columns([3, 1])
                        with image_col:
                            st.image(load_gradcam_display_image(row["file"]), width=GRADCAM_WIDTH)
                        with note_col:
                            render_observation_box(row)

        if not gradcam_selected.empty:
            with st.expander("Selected Grad-CAM example table"):
                st.dataframe(gradcam_selected, hide_index=True, use_container_width=True, height=fit_table_height(gradcam_selected))
        if not gradcam_prediction_table.empty:
            with st.expander("Full Grad-CAM prediction table"):
                gpt_df = gradcam_prediction_table.head(500)
                st.dataframe(gpt_df, hide_index=True, use_container_width=True, height=fit_table_height(gpt_df, max_height=BIG_TABLE_HEIGHT))

elif page == "5.2.6 Setup Check":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("5.2.6 Setup Check")
    st.subheader("App directory")
    st.code(str(APP_DIR))
    st.subheader("Artifact search locations")
    st.code("\n".join(str(p) for p in SEARCH_DIRS))
    st.subheader("Resolved files")
    resolved = pd.DataFrame([{"Artifact": k, "Path": format_path(v)} for k, v in FILES.items()])
    st.dataframe(resolved, hide_index=True, use_container_width=True, height=fit_table_height(resolved))
    st.subheader("Grad-CAM images")
    if gradcam_images:
        gradcam_paths_df = pd.DataFrame({"Path": [str(p) for p in gradcam_images]})
        st.dataframe(gradcam_paths_df, hide_index=True, use_container_width=True, height=fit_table_height(gradcam_paths_df))
    else:
        st.write("No Grad-CAM images found.")
    st.subheader("Prediction columns")
    st.write(list(preds.columns) if not preds.empty else "No predictions loaded")

# ===================================================================
# Chapter 6 — Multimodal
# ===================================================================

elif page == "6. Multimodal":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("6. Multimodal")
    st.write(
        "The multimodal stage combines the best text branch (CamemBERT, full fine-tune) and the best image "
        "branch (ConvNeXt-Base, moderate augmentation, fully unfrozen) to push beyond the unimodal ceilings. "
        "Two fusion strategies were explored: a simple late fusion that requires no additional training, and "
        "an intermediate fusion that trains a projection head on top of the frozen branches."
    )
    st.subheader("Fusion strategies at a glance")
    strat_rows = [
        {
            "Strategy": "Late Fusion (Simple Fusion)",
            "How it works": "Weighted average of the text and image softmax outputs. Only the mixing weight α is tuned on the validation set — no gradient update.",
            "Best Macro F1": f"{mm_late_meta['best_macro_f1']:.4f}" if mm_late_meta.get("best_macro_f1") else "—",
        },
        {
            "Strategy": "Intermediate Fusion",
            "How it works": "Frozen text and image branches feed into a trained projection head and classifier. The fusion layer learns to combine the two feature streams.",
            "Best Macro F1": f"{mm_inter_meta['best_macro_f1']:.4f}" if mm_inter_meta.get("best_macro_f1") else "—",
        },
    ]
    render_html_table(pd.DataFrame(strat_rows))
    st.subheader("Unimodal baselines for reference")
    baseline_rows = [
        {"Model": "CamemBERT (text only)",      "Macro F1": f"{mm_late_meta['f1_text_only']:.4f}"  if mm_late_meta.get("f1_text_only")  else "—"},
        {"Model": "ConvNeXt-Base (image only)",  "Macro F1": f"{mm_late_meta['f1_image_only']:.4f}" if mm_late_meta.get("f1_image_only") else "—"},
    ]
    render_html_table(pd.DataFrame(baseline_rows), max_width="600px")

elif page == "6.1 Overview":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("6.1 Overview Multimodal models")
    st.write(
        "Both fusion models share the same frozen branches. The table below places them next to the best "
        "unimodal baselines to show the gain from combining modalities."
    )

    f1_text  = mm_late_meta.get("f1_text_only")
    f1_image = mm_late_meta.get("f1_image_only")
    late_acc  = mm_late_meta.get("accuracy")
    late_f1   = mm_late_meta.get("best_macro_f1")
    late_wf1  = mm_late_meta.get("weighted_f1")
    inter_f1  = mm_inter_meta.get("best_macro_f1")

    def _f(v, d=4):
        return f"{v:.{d}f}" if v is not None else "—"

    comparison_rows = [
        {
            "Model": "CamemBERT (text only)",
            "Approach": "Text branch alone",
            "Accuracy": "0.8807",
            "Macro F1": "0.8616",
            "Weighted F1": "0.8800",
            "Notes": "Best text-only baseline (T8, 3 epochs, max_len=128)",
        },
        {
            "Model": "ConvNeXt-Base (image only)",
            "Approach": "Image branch alone",
            "Accuracy": "0.720",
            "Macro F1": _f(f1_image),
            "Weighted F1": "0.720",
            "Notes": "Best image-only baseline",
        },
        {
            "Model": "Late Fusion",
            "Approach": "Weighted softmax average (α = 0.55)",
            "Accuracy": _f(late_acc, 3),
            "Macro F1": _f(late_f1),
            "Weighted F1": _f(late_wf1, 4),
            "Notes": "No fusion training; α tuned on val",
        },
        {
            "Model": "Intermediate Fusion",
            "Approach": "Trained projection head on frozen branches",
            "Accuracy": "0.905",
            "Macro F1": _f(inter_f1),
            "Weighted F1": "0.905",
            "Notes": f"Best at epoch {mm_inter_meta.get('best_epoch', '—')}",
        },
    ]
    render_html_table(pd.DataFrame(comparison_rows))

    st.subheader("Key takeaways")
    st.markdown(
        """
        - Both fusion models substantially outperform the image-only baseline (+21 pp macro F1).
        - Late Fusion narrowly beats Intermediate Fusion in macro F1 despite requiring no additional training.
        - CamemBERT carries most of the predictive signal; adding images provides a consistent +1–2 pp gain over text alone.
        - The optimal α of 0.55 (55 % image weight) shows both modalities contribute meaningfully.
        """
    )

elif page == "6.1.1 Simple Fusion":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("6.1.1 Simple Fusion — Late Fusion")
    st.write(
        "Late Fusion is the simplest possible multimodal strategy: the text model and image model each "
        "produce a probability distribution over the 27 product classes independently, and those "
        "distributions are blended by a weighted average. No joint training is needed."
    )

    st.subheader("How it works")
    st.markdown(
        """
        1. **Text branch** — CamemBERT produces softmax probabilities **P_text** (27 classes).
        2. **Image branch** — ConvNeXt-Base produces softmax probabilities **P_image** (27 classes).
        3. **Fusion** — **P_fusion = α · P_image + (1 − α) · P_text**
        4. **Prediction** — argmax(P_fusion)

        The weight α was swept over a grid and the value that maximised macro F1 on the validation set was kept.
        """
    )

    alpha   = mm_late_meta.get("best_alpha")
    macro_f1 = mm_late_meta.get("best_macro_f1")

    if alpha is not None:
        st.subheader("Optimal mixing weight")
        img_pct  = int(round(alpha * 100))
        text_pct = 100 - img_pct
        st.markdown(
            f"**α = {alpha}** → image gets **{img_pct} %**, text gets **{text_pct} %**"
        )

    st.subheader("Validation metrics at optimal α")
    metrics_late = {
        "accuracy":           mm_late_meta.get("accuracy"),
        "macro_f1":           macro_f1,
        "weighted_f1":        mm_late_meta.get("weighted_f1"),
        "validation_samples": 16984,
    }
    render_metric_cards(metrics_late)

    st.subheader("Unimodal vs. fusion comparison")
    def _f(v, d=4):
        return f"{v:.{d}f}" if v is not None else "—"
    comp_df = pd.DataFrame([
        {"Model": "Text only — CamemBERT",        "Macro F1": _f(mm_late_meta.get("f1_text_only"))},
        {"Model": "Image only — ConvNeXt-Base",   "Macro F1": _f(mm_late_meta.get("f1_image_only"))},
        {"Model": f"Late Fusion (α = {alpha})",   "Macro F1": _f(macro_f1)},
    ])
    render_html_table(comp_df, max_width="600px")

elif page == "6.2 Best model — Summary":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("6.2 Best model — Summary")
    st.write(
        "The Late Fusion model (CamemBERT + ConvNeXt-Base, α = 0.55) achieves the highest validation "
        "results across both multimodal strategies and is selected as the final multimodal model."
    )

    alpha    = mm_late_meta.get("best_alpha", "—")
    accuracy = mm_late_meta.get("accuracy")
    macro_f1 = mm_late_meta.get("best_macro_f1")
    wf1      = mm_late_meta.get("weighted_f1")
    f1_text  = mm_late_meta.get("f1_text_only")
    f1_image = mm_late_meta.get("f1_image_only")

    def _f(v, d=3):
        return f"{v:.{d}f}" if v is not None else "—"

    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb; border-radius:0.9rem; padding:1.1rem 1.25rem; background:#fafafa; margin:0.6rem 0 1rem 0;">
          <div style="font-size:1.45rem; font-weight:800; line-height:1.25; color:#262730; margin-bottom:0.25rem;">Late Fusion — CamemBERT + ConvNeXt-Base</div>
          <div style="font-size:1.05rem; color:#4b5563; margin-bottom:0.9rem;">
            Weighted softmax average &nbsp;·&nbsp; α = {alpha} &nbsp;·&nbsp; No additional training required
          </div>
          <div style="display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:0.75rem;">
            <div><div style="font-size:0.82rem; color:#6b7280;">Accuracy</div><div style="font-size:1.55rem; font-weight:700;">{_f(accuracy)}</div></div>
            <div><div style="font-size:0.82rem; color:#6b7280;">Macro F1</div><div style="font-size:1.55rem; font-weight:700;">{_f(macro_f1, 4)}</div></div>
            <div><div style="font-size:0.82rem; color:#6b7280;">Weighted F1</div><div style="font-size:1.55rem; font-weight:700;">{_f(wf1)}</div></div>
            <div><div style="font-size:0.82rem; color:#6b7280;">Image weight α</div><div style="font-size:1.55rem; font-weight:700;">{alpha}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Full model summary")
    summary_rows = pd.DataFrame([
        {"Field": "Text branch",                        "Value": "CamemBERT (full fine-tune, L128)"},
        {"Field": "Image branch",                       "Value": "ConvNeXt-Base (moderate augmentation, fully unfrozen)"},
        {"Field": "Fusion method",                      "Value": "Weighted average of softmax outputs"},
        {"Field": "Optimal α (image weight)",           "Value": str(alpha)},
        {"Field": "Accuracy",                           "Value": _f(accuracy, 4)},
        {"Field": "Macro F1",                           "Value": _f(macro_f1, 4)},
        {"Field": "Weighted F1",                        "Value": _f(wf1, 4)},
        {"Field": "Validation samples",                 "Value": "16 984"},
        {"Field": "Text-only baseline (Macro F1)",      "Value": _f(f1_text, 4)},
        {"Field": "Image-only baseline (Macro F1)",     "Value": _f(f1_image, 4)},
    ])
    render_html_table(summary_rows, max_width="760px")

elif page == "6.2.2 Training history":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("6.2.2 Training history — Intermediate Fusion")
    best_ep = mm_inter_meta.get("best_epoch", "—")
    st.write(
        f"The Intermediate Fusion model was trained for {len(mm_inter_history)} epochs; best validation macro F1 "
        f"was reached at epoch {best_ep}. Late Fusion requires no gradient training — its 'history' is a "
        "single-pass sweep over α values, so only the Intermediate Fusion training curves are shown here."
    )

    if mm_inter_history.empty:
        st.warning("No history.json found for the Intermediate Fusion model.")
    else:
        st.subheader("Training log")
        disp = mm_inter_history.copy()
        for c in disp.select_dtypes("float").columns:
            disp[c] = disp[c].map(lambda x: f"{x:.4f}")
        render_html_table(disp)

        st.subheader("Loss by epoch")
        fig, ax = plt.subplots(figsize=(10.8, 4.4))
        for col, label in [("train_loss", "Train loss"), ("val_loss", "Val loss")]:
            if col in mm_inter_history.columns:
                ax.plot(mm_inter_history["epoch"], mm_inter_history[col], marker="o", label=label)
        if isinstance(best_ep, int):
            ax.axvline(best_ep, color="#ef4444", linestyle="--", alpha=0.55, label=f"Best epoch ({best_ep})")
        ax.set_xlabel("Epoch")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Macro F1 by epoch")
        fig, ax = plt.subplots(figsize=(10.8, 4.4))
        for col, label in [("train_macro_f1", "Train Macro F1"), ("val_macro_f1", "Val Macro F1")]:
            if col in mm_inter_history.columns:
                ax.plot(mm_inter_history["epoch"], mm_inter_history[col], marker="o", label=label)
        if isinstance(best_ep, int):
            ax.axvline(best_ep, color="#ef4444", linestyle="--", alpha=0.55, label=f"Best epoch ({best_ep})")
        ax.set_xlabel("Epoch")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.info(
            f"Val macro F1 peaks at epoch {best_ep} and degrades slightly afterwards, indicating mild "
            "overfitting of the fusion head while the branch weights remain frozen."
        )

elif page == "6.2.3 Classification Report + Confusion Matrix":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("6.2.3 Classification Report — Late Fusion")

    if not mm_late_report:
        st.warning("No classification report found for the Late Fusion model.")
    else:
        mm_metrics = parse_report_metrics(mm_late_report)
        if mm_late_meta.get("accuracy"):
            mm_metrics["accuracy"] = mm_late_meta["accuracy"]
        if mm_late_meta.get("best_macro_f1"):
            mm_metrics["macro_f1"] = mm_late_meta["best_macro_f1"]
        if mm_late_meta.get("weighted_f1"):
            mm_metrics["weighted_f1"] = mm_late_meta["weighted_f1"]
        mm_metrics.setdefault("validation_samples", 16984)

        st.subheader("Main validation metrics")
        render_metric_cards(mm_metrics)

        class_df, summary_df = parse_classification_report_table(mm_late_report, id2label)
        if not class_df.empty:
            st.subheader("Per-class classification report")
            class_disp = class_df.copy()
            for c in ["Precision", "Recall", "F1-score"]:
                class_disp[c] = class_disp[c].map(lambda x: f"{x:.2f}")
            render_html_table(class_disp, max_width="1050px", compact=True)
        if not summary_df.empty:
            st.subheader("Summary rows")
            sum_disp = summary_df.copy()
            for c in ["Precision", "Recall", "F1-score / score"]:
                sum_disp[c] = sum_disp[c].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            render_html_table(sum_disp, max_width="900px")

    st.subheader("Confusion matrix")
    if mm_late_cm_png:
        st.image(str(mm_late_cm_png), caption="Normalised confusion matrix — Late Fusion (α = 0.55)", use_container_width=True)
    else:
        st.warning("confusion_matrix.png not found in the Late Fusion data folder.")

elif page == "6.2.5 Error Analysis":
    st.title("Rakuten Multimodal Product Data Classification")
    st.header("6.2.5 Error Analysis — Late Fusion")

    if mm_late_preds.empty:
        st.warning("No val_predictions.csv found for the Late Fusion model.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        total   = len(mm_late_preds)
        correct = int(mm_late_preds["correct"].sum()) if "correct" in mm_late_preds.columns else 0
        c1.metric("Validation rows", f"{total:,}")
        c2.metric("Correct",         f"{correct:,}")
        c3.metric("Errors",          f"{total - correct:,}")
        c4.metric("Mean confidence",
                  f"{mm_late_preds['confidence'].mean():.3f}"
                  if "confidence" in mm_late_preds.columns else "—")

        filtered = mm_late_preds.copy()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            view = st.selectbox("Prediction status", ["All", "Only errors", "Only correct"])
            if "correct" in filtered.columns:
                if view == "Only errors":
                    filtered = filtered[~filtered["correct"]]
                elif view == "Only correct":
                    filtered = filtered[filtered["correct"]]
        with col_b:
            if "true_label" in filtered.columns:
                classes = ["All"] + sorted(
                    filtered["true_label"].astype(str).unique().tolist(),
                    key=lambda x: int(x) if x.isdigit() else x,
                )
                true_choice = st.selectbox("True class", classes)
                if true_choice != "All":
                    filtered = filtered[filtered["true_label"].astype(str) == true_choice]
        with col_c:
            if "pred_label" in filtered.columns:
                classes = ["All"] + sorted(
                    filtered["pred_label"].astype(str).unique().tolist(),
                    key=lambda x: int(x) if x.isdigit() else x,
                )
                pred_choice = st.selectbox("Predicted class", classes)
                if pred_choice != "All":
                    filtered = filtered[filtered["pred_label"].astype(str) == pred_choice]

        if "confidence" in filtered.columns and not filtered.empty:
            conf_range = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), 0.01)
            mask = (
                (filtered["confidence"] >= conf_range[0]) &
                (filtered["confidence"] <= conf_range[1])
            ).fillna(False)
            filtered = filtered[mask]

        sort_options = [
            c for c in ["confidence", "true_label", "pred_label", "productid", "imageid"]
            if c in filtered.columns
        ]
        if sort_options:
            sort_col  = st.selectbox("Sort by", sort_options)
            ascending = st.checkbox("Ascending", value=False)
            filtered  = filtered.sort_values(sort_col, ascending=ascending)

        st.subheader("Filtered validation predictions")
        st.dataframe(
            filtered, hide_index=True, use_container_width=True,
            height=fit_table_height(filtered, max_height=BIG_TABLE_HEIGHT),
        )
        st.download_button(
            "Download filtered rows as CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            "filtered_late_fusion_predictions.csv",
            "text/csv",
        )
