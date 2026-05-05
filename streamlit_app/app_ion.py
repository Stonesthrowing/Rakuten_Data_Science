import streamlit as st

from pathlib import Path
import pandas as pd

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import re, html, nltk
from nltk.corpus import stopwords

@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return (
        set(stopwords.words("french")) |
        set(stopwords.words("english")) |
        set(stopwords.words("german"))
    )

stopword_set = load_stopwords()

def remove_repeated_blocks(text, min_block_len=100):
    if not isinstance(text, str):
        return text
    text = text.strip()
    if len(text) < 2 * min_block_len:
        return text
    pattern = re.compile(r"(.{" + str(min_block_len) + r",}?)(?:\1)+", re.DOTALL)
    prev = None
    while text != prev:
        prev = text
        text = pattern.sub(r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_and_remove_stopwords(text):
    text = "" if not isinstance(text, str) else text
    text = html.unescape(text)
    text = re.sub(r"<.*?>", " ", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in stopword_set and len(t) > 1]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()

APP_DIR = Path(__file__).resolve()
ROOT_DIR = APP_DIR.parent

IMAGE_DIR = ROOT_DIR / "streamlit_app" / "images"
X_TEST_PATH = ROOT_DIR / "data" / "raw" / "test_clean.csv"
IMG_DIR = ROOT_DIR / "data" / "raw" / "images" / "image_test"
MODEL_DIR = ROOT_DIR / "data" / "models" / "camembert_run4"

label_names = {
    2583: "swimming pool",
    1560: "furniture",
    1300: "gadgets",
    2060: "deco",
    2522: "office products",
    1280: "toys",
    2403: "literature/media",
    2280: "journals",
    1920: "home textiles",
    1160: "collection cards",
    1320: "baby products",
    10: "books",
    2705: "books French",
    1140: "toys",
    2582: "garden items",
    40: "disks/games media",
    2585: "tools",
    1302: "toys",
    1281: "toys",
    50: "gaming products",
    2462: "gaming consoles/disks",
    2905: "pc games",
    60: "gaming consoles",
    2220: "pets",
    1301: "socks",
    1940: "plastic pouch",
    1180: "collectible items"
}

@st.cache_resource
def load_camembert_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    with open(model_dir / "id2label.json", "r", encoding="utf-8") as f:
        id2label = json.load(f)

    return tokenizer, model, id2label


def predict_top3(text, model_dir):
    tokenizer, model, id2label = load_camembert_model(model_dir)

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    top_probs, top_ids = torch.topk(probs, k=3)

    results = []
    for i in range(3):
        class_id = str(top_ids[i].item())
        prob = top_probs[i].item()

        code = int(id2label[class_id])
        name = label_names.get(code, "unknown")

        results.append({
            "Rank": int(i + 1),
            "Predicted code": int(code),
            "Label": name,
            "Probability": f"{prob * 100:.2f}%"
        })

    return pd.DataFrame(results)


def display_formatted_df(df):
    config = {}

    for col in df.columns:
        if col in ["Rank", "Predicted code"]:
            config[col] = st.column_config.NumberColumn(format="%d")
        elif df[col].dtype in ["float64", "float32"]:
            config[col] = st.column_config.NumberColumn(format="%.3f")

    st.dataframe(
        df,
        column_config=config,
        use_container_width=False,
        hide_index=True
    )

@st.cache_data
def load_train_data():
    return pd.read_csv(ROOT_DIR / "train_clean.csv")

@st.cache_data
def get_dataset_examples(random_state=22):
    df = load_train_data()
    return df.sample(5, random_state=random_state)


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Rakuten Product Classification",
    page_icon="🛒",
    layout="wide",
)


# =========================
# Sidebar navigation
# =========================
st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Data Exploration",
        "Preprocessing",
        "Modeling",
        "Live Demo",
        "Conclusion",
    ],
)


# =========================
# Header
# =========================
st.title("Rakuten Product Classification")

# =========================
# Overview
# =========================
if section == "Overview":
    st.header("Overview")

    st.write(
        """
        This project focuses on classifying Rakuten marketplace products into one of 27
        categories (`prdtypecode`) using textual and visual information.
        """
    )

    # --- Key metrics ---
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Categories", "27")
    col2.metric("Train samples", "84 916")
    col3.metric("Test samples", "13 812")
    col4.metric("Missing descriptions", "~35%")

    # --- Dataset description + examples ---
    st.subheader("Dataset structure")

    st.write(
        """
        Each product is described by four fields
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.write("- **designation**: product title")
        st.write("- **image**: product image")

    with c2:
        st.write("- **description**: detailed text")
        st.write("- **prdtypecode**: category label")

    st.info(
        "The objective is to predict the product category (`prdtypecode`) for unseen test products using the available text and image information."
    )

    

    # --- Multimodal aspect ---
    st.subheader("Multimodal data")

    st.write(
        """
        The dataset combines:
        - textual information (title and description)
        - visual information (product images)

        Images are stored separately and linked using `imageid` and `productid`.
        """
    )

    # --- Challenges ---
    st.subheader("Key challenges")

    st.write(
        """
        - Missing descriptions (~35%) → models must rely heavily on titles  
        - 27 classes → multi-class classification problem  
        - Heterogeneous products → high variability in text and images  
        - Some categories are visually similar but textually distinct (and vice versa)
        """
    )

# =========================
# Data Exploration
# =========================
elif section == "Data Exploration":
    sub_section = st.sidebar.radio(
        "Explore",
        ["Text", "Images"]
    )
    if sub_section == "Text":
        st.header("Text Exploration")
     #   st.header("Data Exploration")

        st.write(
            """
            Key dataset characteristics that influenced preprocessing, modeling, and evaluation.
            """
        )

        # =========================
        # Category distribution
        # =========================
        st.subheader("Category distribution")

        col1, col2, col3 = st.columns(3)

        col1.metric("Categories", "27")
        col2.metric("Largest class", ">10 000")
        col3.metric("Smallest classes", "~700–800")

        st.write(
            """
            - Moderate class imbalance across product categories
            - Even the smallest classes contain several hundred samples
            - Macro F1 is important because accuracy alone can hide weak minority-class performance
            """
        )

        IMAGE_PATH = IMAGE_DIR / "category_balance.png"

        st.image(
            str(IMAGE_PATH),
            caption="Category distribution: largest vs smallest classes",
            width="stretch"
        )

        # =========================
        # Text fields overview
        # =========================
        st.subheader("Text fields")

        col1, col2 = st.columns(2)

        with col1:
            st.write(
                """
                **Designation**
                - Product title
                - Always available
                - Short and consistent
                - Strong category signal
                """
            )

        with col2:
            st.write(
                """
                **Description**
                - ~35% missing
                - Longer and more variable
                - Adds product attributes and context
                """
            )

        # =========================
        # Text length statistics
        # =========================
        st.subheader("Text length")

        col1, col2 = st.columns(2)

        col1.metric("Avg title length", "~11 words")
        col2.metric("Avg description length", "~95 words")

        st.write(
            """
            - Titles are compact and stable
            - Descriptions are longer, skewed, and sometimes noisy
            - Very long descriptions often contain duplicated content
            """
        )

        IMAGE_PATH = IMAGE_DIR / "text_length.png"

        st.image(
            str(IMAGE_PATH),
            caption="Titles are short and consistent; descriptions are longer, variable, and often missing",
            width="stretch"
        )

        # =========================
        # Data quality issues
        # =========================
        st.subheader("Data quality")

        col1, col2, col3 = st.columns(3)

        col1.metric("Missing descriptions", "~35%")
        col2.metric("Duplicated text blocks", "~1.5%")
        col3.metric("Numeric tokens", "~8–9%")

        st.write(
            """
            Duplicated description segments were removed to reduce noise while preserving the majority of samples.
            Numeric tokens were kept because product references, sizes, and model numbers may be useful.
            """
        )

        # =========================
        # Vocabulary insights
        # =========================
        st.subheader("Vocabulary insights")

        col1, col2 = st.columns(2)

        col1.metric("Title vocabulary", "~82k tokens")
        col2.metric("Description vocabulary", "~137k tokens")

        st.write(
            """
            - Titles often contain category-defining product keywords
            - Descriptions mostly add attributes such as size, color, material, and condition
            """
        )

        IMAGE_PATH = IMAGE_DIR / "token_comparison.png"

        st.image(
            str(IMAGE_PATH),
            caption="Titles contain category-defining keywords, while descriptions add broader and often less specific vocabulary (blue = shared, red = description-only)",
            width="stretch"
        )

        # =========================
        # Key takeaways
        # =========================
        st.subheader("Key takeaways")

        st.success(
            """
            Titles are the strongest text feature.  \n
            Descriptions provide useful but noisier context.  \n
            Class imbalance makes macro F1 more informative than accuracy alone.
            """
        )

    elif sub_section == "Images":
        st.header("Images Exploration")

        # =========================
        # Dataset-level stats
        # =========================
        st.subheader("Dataset overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total images", "98 728")
        col2.metric("Train images", "84 916")
        col3.metric("Test images", "13 812")

        col4, col5 = st.columns(2)

        col4.metric("Disk size", "2.44 GB")
        col5.metric("Missing images", "0 %")

        st.write(
            """
            Each product is associated with one image, linked via `imageid` and `productid`.
            """
        )

        # =========================
        # Image properties
        # =========================
        st.subheader("Image properties")

        st.write(
            """
            - format: JPG (standardized across dataset)  
            - resolution: 500 × 500 pixels  
            - color depth: 24-bit  
            - resolution: 96 dpi  
            """
        )

        st.write(
            """
            All images follow a consistent format and size.
            """
        )

        # =========================
        # Data quality observations
        # =========================
        st.subheader("Data quality")

        st.write(
            """
            - images are of good quality  
            - certain categories are visually similar  
            """
        )

        # =========================
        # Example samples
        # =========================
        st.subheader("Sample products")

        sample = get_dataset_examples()

        for _, row in sample.iterrows():
            image_path = IMAGE_DIR / f"image_{row['imageid']}_product_{row['productid']}.jpg"

            col_img, col_text = st.columns([1, 2])

            with col_img:
                if image_path.exists():
                    st.image(str(image_path), width="stretch")
                else:
                    st.caption("Image missing")

            with col_text:
                st.write(f"**Category:** `{row['prdtypecode']}`")
                st.write(f"**Designation:**  \n {row['designation']}")

                desc = row.get("description", "")
                if pd.isna(desc) or str(desc).strip() == "":
                    desc = "Missing"
                else:
                    desc = str(desc)[:500] + "..."

                st.write(f"**Description:**  \n {desc}")

            st.divider()

        # =========================
        # Key takeaway
        # =========================
        st.subheader("Key takeaway")

        st.write(
            """
            Images provide complementary information to text, but their variability and
            inconsistent quality make standalone image classification more challenging.
            """
        )

# =========================
# Preprocessing
# =========================
elif section == "Preprocessing":
    st.header("Text Preprocessing")

    st.write(
        """
        Text preprocessing prepares raw product text for different modeling approaches.
        The pipeline is designed to clean noise while preserving informative signals.
        """
    )

    # =========================
    # Core cleaning steps
    # =========================
    st.subheader("Cleaning pipeline")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        - lowercase conversion  
        - removal of punctuation and special characters  
        """)

    with col2:
        st.markdown("""
        - removal of HTML tags and encoded text  
        - whitespace normalization  
        """)

    with col3:
        st.markdown("""
        - removal of short tokens (<2 characters)  
        - stopword removal (French, English, German)  
        """)

    # =========================
    # Handling missing data
    # =========================
    st.subheader("Handling missing data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - product titles are always available  
        - descriptions are missing in ~35% of cases  
        """)

    with col2:
        st.markdown("""
        - models must remain robust when description is absent  
        """)

    # =========================
    # Numeric tokens
    # =========================
    st.subheader("Numeric information")

    col_text, col_plot = st.columns([1.6, 0.9])  # narrower plot

    with col_text:
        st.markdown(
            """
            - ~8–9% of tokens are purely numeric  
            - numbers may encode useful information (size, quantity, model IDs)  
            """
        )

        st.markdown(
            """
            Two configurations are evaluated:
            - keep numeric tokens  
            - remove numeric tokens  

            This allows measuring their impact on model performance.
            """
        )

    with col_plot:
        #st.markdown("<br><br>", unsafe_allow_html=True)  # vertical alignment tweak
        
        IMAGE_PATH = IMAGE_DIR / "numeric_token_share.png"

        st.image(
            str(IMAGE_PATH),
            caption="Share of numeric tokens in titles and descriptions.",
            width=380  # key change: smaller image
        )

    # =========================
    # Deduplication
    # =========================
    st.subheader("Description deduplication")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ~1.5% of descriptions contain repeated text blocks that artificially inflate length.
        """)

    with col2:
        st.markdown("""
        A preprocessing step removes consecutive duplicated segments while preserving
        the original content.
        """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        - reduces extreme outliers  
        """)

    with col2:
        st.markdown("""
        - does not affect the majority of samples  
        """)

    with col3:
        st.markdown("""
        - improves overall data quality  
        """)

    # =========================
    # Tokenization
    # =========================
    st.subheader("Tokenization")

    col1, col2 = st.columns([1.0, 1.35])

    with col1:
            st.markdown(
            """
            After cleaning, text is tokenized to enable:
            - vocabulary construction  
            - frequency-based representations (TF-IDF)  
            - input formatting for neural models  
            """
        )


    with col2:

        IMAGE_PATH = IMAGE_DIR / "tokenization_example.png"

        st.image(
            str(IMAGE_PATH),
            caption="Example transformation from raw product title to tokenized input.",
            width="stretch"
            )

    # =========================
    # Model-specific preprocessing
    # =========================
    st.subheader("Model-specific processing")

    st.write(
        """
        Different models require different preprocessing strategies:
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        - **TF-IDF models**  
        use cleaned and tokenized text with sparse vectorization  
        """)

    with col2:
        st.markdown("""
        - **Embedding-based models**  
        rely on tokenized sequences  
        """)

    with col3:
        st.markdown("""
        - **CamemBERT**  
        uses its own tokenizer and subword encoding  
        (minimal manual preprocessing required)  
        """)

    # =========================
    # Key takeaway
    # =========================
    st.subheader("Key takeaway")

    st.success(
        """
        Preprocessing removes noise while preserving informative signals.
        """
    )


# =========================
# Modeling
# =========================
elif section == "Modeling":
    sub_section = st.sidebar.radio(
        "Explore",
        [
            "Text classification",
            "Image classification",
            "Multimodal classification",
        ]
    )
    st.header("Modeling")

    if sub_section == "Text classification":
        st.subheader("Text Classification")

        text_modeling_section = st.sidebar.radio(
            "Text modeling section",
            [
                "Overview",
                "Best model",
            ]
        )

        if text_modeling_section == "Overview":

            st.subheader("Text modeling overview")

            st.caption("Establishing a strong text-only baseline before multimodal models.")

            # =========================
            # Main questions
            # =========================
            st.subheader("Key questions")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                    - which text source matters most?  
                    - do bigrams improve performance?  
                    """
                )

            with col2:
                st.markdown(
                    """
                    - should numeric tokens be kept?  
                    - which classifier works best?  
                    """
                )

            # =========================
            # Representation results
            # =========================
            st.subheader("Text representation")

            representation_results = pd.DataFrame(
                {
                    "Text column": [
                        "text_combined",
                        "text_combined",
                        "designation",
                        "designation_nodigits",
                        "description_dedup",
                    ],
                    "TF-IDF config": [
                        "bigram",
                        "unigram",
                        "bigram",
                        "bigram",
                        "bigram",
                    ],
                    "Accuracy": [0.8412, 0.8340, 0.8281, 0.8138, 0.6070],
                    "F1 weighted": [0.8398, 0.8327, 0.8272, 0.8125, 0.6175],
                    "F1 macro": [0.8284, 0.8210, 0.8128, 0.7976, 0.6064],
                }
            )

            display_formatted_df(representation_results)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                    **Best setup**
                    - title + description  
                    - TF-IDF bigrams  
                    - numeric tokens kept  
                    """
                )

            with col2:
                st.markdown(
                    """
                    **Key insight**
                    - combining fields improves signal  
                    - removing digits hurts performance  
                    """
                )

            # =========================
            # Classifier results
            # =========================
            st.subheader("Classifier comparison")

            classifier_results = pd.DataFrame(
                {
                    "Classifier": [
                        "LinearSVC",
                        "PassiveAggressive",
                        "LogisticRegression",
                        "ComplementNB",
                        "SGDClassifier",
                        "MultinomialNB",
                    ],
                    "Accuracy": [0.8412, 0.8282, 0.8173, 0.7905, 0.7140, 0.7068],
                    "F1 weighted": [0.8398, 0.8269, 0.8171, 0.7801, 0.7007, 0.6851],
                    "F1 macro": [0.8284, 0.8136, 0.7995, 0.7645, 0.6500, 0.6450],
                }
            )
            
            display_formatted_df(classifier_results)
                                 
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                    **Best model**
                    - LinearSVC  
                    - handles high-dimensional sparse data well  
                    """
                )

            with col2:
                st.markdown(
                    """
                    **Observation**
                    - macro F1 < weighted F1  
                    - some classes remain harder  
                    """
                )

            # =========================
            # Final TF-IDF setup
            # =========================
            st.subheader("Baseline configuration")

            col1, col2, col3 = st.columns(3)

            col1.metric("Input", "Title + description")
            col2.metric("Vectorization", "TF-IDF bigram")
            col3.metric("Classifier", "LinearSVC")

            st.success("Strong, fast, and interpretable baseline.")

            # =========================
            # Key findings
            # =========================
            st.subheader("Key findings")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                    - combining text fields helps  
                    - bigrams > unigrams  
                    - numeric tokens are useful  
                    """
                )

            with col2:
                st.markdown(
                    """
                    - LinearSVC performs best  
                    - short texts are harder  
                    - errors in similar categories  
                    """
                )

            # =========================
            # Beyond TF-IDF
            # =========================
            st.subheader("Beyond TF-IDF")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    """
                    **TF-IDF**
                    - sparse  
                    - keyword-based  
                    """
                )

            with col2:
                st.markdown(
                    """
                    **Embeddings**
                    - dense  
                    - semantic  
                    """
                )

            with col3:
                st.markdown(
                    """
                    **Transformers**
                    - contextual  
                    - task-adaptive  
                    """
                )

            # =========================
            # Sentence embeddings
            # =========================
            st.subheader("Sentence embeddings")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Accuracy", "0.71")
                st.metric("Macro F1", "0.68")

            with col2:
                st.markdown(
                    """
                    - lose lexical precision  
                    - weaker on keyword-driven tasks  
                    """
                )

            st.info("Semantic compression reduces performance for product classification.")

            # =========================
            # Model comparison
            # =========================
            st.subheader("Model comparison")

            st.image(
                str(IMAGE_DIR / "model_comparison.png"),
                caption="CamemBERT slightly outperforms TF-IDF, while embeddings lag behind.",
                use_container_width=True
            )
            
        elif text_modeling_section == "Best model":

            st.subheader("Best text model: CamemBERT")

            # =========================
            # Model summary
            # =========================
            col1, col2, col3 = st.columns(3)
            col1.metric("Model", "CamemBERT")
            col2.metric("Input", "Title + description")
            col3.metric("Type", "Transformer")

            st.caption("Captures contextual relationships between words through fine-tuning.")

            # =========================
            # Final performance
            # =========================
            st.subheader("Final performance")

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", "0.8719")
            col2.metric("Macro F1", "0.8557")
            col3.metric("Rank", "Best")

            st.success("Best-performing text model across all evaluated approaches.")

            # =========================
            # Model comparison
            # =========================
            st.subheader("Model comparison")

            st.image(
                str(IMAGE_DIR / "model_comparison.png"),
                caption="CamemBERT slightly outperforms TF-IDF, while MiniLM-based models lag behind.",
                use_container_width=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    """
        **TF-IDF**
        - strong keyword matching  
        - competitive baseline  
        """
                )

            with col2:
                st.markdown(
                    """
        **MiniLM**
        - loses lexical detail  
        - weakest performance  
        """
                )

            with col3:
                st.markdown(
                    """
        **CamemBERT**
        - context-aware  
        - best overall performance  
        """
                )

            # =========================
            # Key training decisions
            # =========================
            st.subheader("Key training decisions")

            col1, col2, col3 = st.columns(3)

            col1.metric("Max length", "256")
            col2.metric("Epochs", "4")
            col3.metric("Trend", "Improving")

            st.markdown(
                """
        - longer sequences improve performance  
        - gains continue up to 4 epochs  
        - best configuration: **256 / 4 epochs**  
        """
            )

            # =========================
            # Training behavior
            # =========================
            st.subheader("Training behavior")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
        - training loss ↓ steadily  
        - validation improves up to epoch 4  
        - slight overfitting appears after  
        """
                )

            with col2:
                st.image(
                    str(IMAGE_DIR / "camembert_training_curves_split.png"),
                    caption="Performance stabilizes around epochs 3–4.",
                    use_container_width=True
                )

            # =========================
            # Why it works
            # =========================
            st.subheader("Why it works")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
        **Strengths**
        - captures context  
        - uses full sequences  
        - adapts via fine-tuning  
        """
                )

            with col2:
                st.markdown(
                    """
        **Compared to TF-IDF**
        - not limited to keywords  
        - understands phrasing  
        """
                )

            # =========================
            # Per-class performance
            # =========================
            st.subheader("Per-class performance")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Categories improved", "17 / 27")

            with col2:
                st.markdown(
                    """
        TF-IDF remains competitive in keyword-driven categories.
        """
                )

            st.image(
                str(IMAGE_DIR / "per_class_f1_delta_tfidf_camembert_top_changes.png"),
                caption="Per-class performance differences.",
                use_container_width=True
            )

            # =========================
            # Where models differ
            # =========================
            st.subheader("Where models differ")

            st.image(
                str(IMAGE_DIR / "bow_class_comparison.png"),
                caption="Red: TF-IDF better | Blue: CamemBERT better.",
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
        🔴 **TF-IDF**
        - repetitive keywords  
        - strong lexical signals  
        - term-driven categories  
        """
                )

            with col2:
                st.markdown(
                    """
        🔵 **CamemBERT**
        - diverse language  
        - distributed meaning  
        - context-dependent  
        """
                )

            st.info("TF-IDF = keywords | CamemBERT = context → complementary strengths")

            # =========================
            # Limitations & transition
            # =========================
            st.subheader("Limitations")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
        - overlapping vocabulary across classes  
        - text alone not always sufficient  
        """
                )

            with col2:
                st.markdown(
                    """
        Visual features (shape, color, appearance) can improve classification.
        """
                )

            st.success("Next step: evaluate image-based models.")

        elif modeling_choice == "Image classification":
            st.subheader("Image Classification")
            st.info("Placeholder for teammate image-model section.")

        elif modeling_choice == "Multimodal classification":
            st.subheader("Multimodal Classification")
            st.info("Placeholder for teammate multimodal-model section.")
    elif sub_section == "Image classification" :
        st.header("Artur and Felix")
    elif sub_section == "Multimodal classification" :
        st.header("Sümeyra")


# =========================
# Live Demo
# =========================
elif section == "Live Demo":
    st.header("Live Demo")

    st.subheader("1. Random product from test set")

    st.write(
        """
        A random product is selected from the test split.
        The app shows the product text, its image, and the top predicted categories.
        """
    )

    @st.cache_data
    def load_x_test(path):
        return pd.read_csv(path)

    X_test = load_x_test(X_TEST_PATH)

    if "demo_row" not in st.session_state:
        st.session_state.demo_row = None

    if st.button("Show random product"):
        st.session_state.demo_row = X_test.sample(1).iloc[0]

    if st.session_state.demo_row is not None:
        row = st.session_state.demo_row

        st.markdown("#### Product information")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("**Designation**")
            st.write(row["designation"])

            description = row["description"]

            if pd.isna(description) or str(description).strip() == "":
                description = "Missing"

            st.write("**Description**")
            st.write(description)

        with col2:
            image_path = IMG_DIR / f"image_{row['imageid']}_product_{row['productid']}.jpg"

            if image_path.exists():
                st.image(
                    str(image_path),
                    caption="Product image",
                    width=300
                )
            else:
                st.warning("Image not found")

        st.markdown("#### Text model prediction")

        # combine text exactly as in text modeling
        text_input = str(row["designation_clean"]) + " " + ("" if description == "Missing" else str(row["description_clean"]))

        if st.button("Predict with CamemBERT"):
            pred_df = predict_top3(text_input, MODEL_DIR)

            display_formatted_df(pred_df)

           
    st.divider()

    st.subheader("2. User input")

    user_designation = st.text_input("Product title / designation")
    user_description = st.text_area("Product description (optional)")

    if st.button("Predict user input"):
        if user_designation.strip() == "":
            st.warning("Please enter a product title.")
        else:        
            text_input = normalize_and_remove_stopwords(user_designation.strip())

            if user_description.strip() != "":            
                text_input_a = remove_repeated_blocks(user_description.strip())
                text_input_b = normalize_and_remove_stopwords(text_input_a)
                text_input += " " + text_input_b

            pred_df = predict_top3(text_input, MODEL_DIR)

            st.markdown("#### Predicted category")

            display_formatted_df(pred_df)


# =========================
# Conclusion
# =========================
elif section == "Conclusion":
    st.header("Conclusion")

    st.write(
        """
        The project demonstrates that text-based models provide a strong baseline for
        product classification. Classical approaches are efficient and competitive,
        while transformer-based models achieve the best performance.
        """
    )

    st.subheader("Main takeaway")

    st.write(
        """
        Text information is highly informative, but certain product categories benefit
        from visual context. This supports the use of multimodal models for the final system.
        """
    )