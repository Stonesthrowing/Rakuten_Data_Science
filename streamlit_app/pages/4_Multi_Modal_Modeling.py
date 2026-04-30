import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Multimodal Modeling", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent

st.title("🔀 Multimodal Modeling")
st.markdown("### Multimodal experiments, fusion strategies, and model comparison")

MODEL_INFO = {
    "clip_text_vision_frozen": {
        "label": "CLIP Text + CLIP Vision (Frozen)",
        "description": (
            "This model uses CLIP text and CLIP vision encoders as frozen backbones. "
            "Only the fusion/classification head is trained. "
            "The goal of this setup is to test whether pretrained multimodal representations "
            "already provide enough discriminative information without updating the encoder weights."
        ),
        "modalities": "Text + Image",
        "text_backbone": "CLIP Text Encoder",
        "image_backbone": "CLIP Vision Encoder",
        "training_strategy": "Frozen encoders",
        "run_name": "clip_multimodal_m3_fixed_v2",
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
    },
    "clip_text_vision_staged": {
        "label": "CLIP Text + CLIP Vision (Staged / Partial Unfreeze)",
        "description": (
            "This model starts from pretrained CLIP encoders and applies staged training. "
            "The classification head is trained first, then selected upper layers are unfrozen "
            "to adapt the model more closely to the Rakuten classification task."
        ),
        "modalities": "Text + Image",
        "text_backbone": "CLIP Text Encoder",
        "image_backbone": "CLIP Vision Encoder",
        "training_strategy": "Staged / partial unfreeze",
        "run_name": "clip_multimodal_m3_partial_unfreeze_v2",
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
    },
    "camembert_clipvision_frozen": {
        "label": "CamemBERT + CLIP Vision (Frozen)",
        "description": (
            "This model combines CamemBERT text representations with CLIP vision features. "
            "In the frozen setup, both backbones are kept fixed and only the fusion/classification layers are trained. "
            "This allows evaluation of multimodal complementarity without end-to-end fine-tuning."
        ),
        "modalities": "Text + Image",
        "text_backbone": "CamemBERT",
        "image_backbone": "CLIP Vision Encoder",
        "training_strategy": "Frozen encoders",
        "run_name": "mm_camembert_clip_gated_fusion_frozen",
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
    },
    "camembert_clipvision_staged": {
        "label": "CamemBERT + CLIP Vision (Staged / Partial Unfreeze)",
        "description": (
            "This multimodal model combines CamemBERT and CLIP Vision in a staged training setup. "
            "After training the fusion/classification layers, selected encoder layers are unfrozen. "
            "This allows better task adaptation while keeping training more stable than full end-to-end optimization."
        ),
        "modalities": "Text + Image",
        "text_backbone": "CamemBERT",
        "image_backbone": "CLIP Vision Encoder",
        "training_strategy": "Staged / partial unfreeze",
        "run_name": "mm_camembert_clip_gated_fusion_staged_unfreeze",
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
    },
    "late_fusion": {
        "label": "Late Fusion",
        "description": (
            "This approach combines prediction outputs from separate text and image models. "
            "Instead of learning a shared feature representation, it fuses the final probabilities or logits. "
            "This is a strong practical baseline because it is simple, modular, and easy to compare against end-to-end fusion models."
        ),
        "modalities": "Text + Image",
        "text_backbone": "Separate text model",
        "image_backbone": "Separate image model",
        "training_strategy": "Output-level fusion",
        "run_name": "late_fusion_v1",
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
    },
}

def get_macro_f1_plot_path(run_name: str) -> Path:
    return BASE_DIR / "outputs" / "image_modeling" / run_name / "macro-f1.png"

selected_key = st.selectbox(
    "Select multimodal model",
    options=list(MODEL_INFO.keys()),
    format_func=lambda key: MODEL_INFO[key]["label"]
)

model_data = MODEL_INFO[selected_key]
plot_path = get_macro_f1_plot_path(model_data["run_name"])

left_col, right_col = st.columns([1.15, 1], gap="large")

with left_col:
    st.markdown(f"## {model_data['label']}")
    st.write(model_data["description"])

    st.markdown("### Model Summary")
    st.write(f"**Modalities:** {model_data['modalities']}")
    st.write(f"**Text Backbone:** {model_data['text_backbone']}")
    st.write(f"**Image Backbone:** {model_data['image_backbone']}")
    st.write(f"**Training Strategy:** {model_data['training_strategy']}")
    st.write(f"**Run Name:** `{model_data['run_name']}`")

with right_col:
    st.markdown("### Performance")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Accuracy",
            "-" if model_data["accuracy"] is None else f"{model_data['accuracy']:.4f}"
        )
    with m2:
        st.metric(
            "Macro F1",
            "-" if model_data["macro_f1"] is None else f"{model_data['macro_f1']:.4f}"
        )
    with m3:
        st.metric(
            "Weighted F1",
            "-" if model_data["weighted_f1"] is None else f"{model_data['weighted_f1']:.4f}"
        )

st.markdown("## Training Curve")

if plot_path.exists():
    st.image(str(plot_path), caption=f"Macro F1 curve — {model_data['run_name']}", use_container_width=True)
else:
    st.warning(f"macro-f1.png not found for run: {model_data['run_name']}")
    st.code(str(plot_path))

st.markdown("## Interpretation")

if selected_key == "clip_text_vision_frozen":
    st.info(
        "This model tests the pure transfer capability of CLIP as a frozen multimodal backbone. "
        "It is useful as a reference point before staged fine-tuning."
    )

elif selected_key == "clip_text_vision_staged":
    st.info(
        "This model evaluates whether partial unfreezing improves the frozen CLIP baseline "
        "by adapting higher-level multimodal features to the Rakuten dataset."
    )

elif selected_key == "camembert_clipvision_frozen":
    st.info(
        "This setup combines a strong French text encoder with visual features while keeping both backbones fixed. "
        "It isolates the value of fusion without updating the pretrained encoders."
    )

elif selected_key == "camembert_clipvision_staged":
    st.info(
        "This model allows controlled adaptation of both text and vision representations. "
        "It is especially useful when the frozen setup is too rigid and the task requires domain adaptation."
    )

elif selected_key == "late_fusion":
    st.info(
        "Late fusion is an important multimodal baseline because it is simple, interpretable, and modular. "
        "It helps determine whether explicit joint feature learning really improves over combining already strong unimodal predictors."
    )

st.markdown("## All Multimodal Models")

comparison_rows = []
for key, info in MODEL_INFO.items():
    comparison_rows.append({
        "Model": info["label"],
        "Text Backbone": info["text_backbone"],
        "Image Backbone": info["image_backbone"],
        "Training Strategy": info["training_strategy"],
        "Accuracy": "-" if info["accuracy"] is None else f"{info['accuracy']:.4f}",
        "Macro F1": "-" if info["macro_f1"] is None else f"{info['macro_f1']:.4f}",
        "Weighted F1": "-" if info["weighted_f1"] is None else f"{info['weighted_f1']:.4f}",
        "Run Name": info["run_name"],
    })

st.dataframe(comparison_rows, use_container_width=True)