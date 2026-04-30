import streamlit as st

st.set_page_config(
    page_title="Rakuten Multimodal Product Classification",
    page_icon="🛍️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-box {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        border: 1px solid rgba(120, 120, 120, 0.18);
        background: rgba(250, 250, 250, 0.03);
        margin-bottom: 1rem;
    }
    .card-box {
        padding: 1.1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(120, 120, 120, 0.18);
        background: rgba(250, 250, 250, 0.02);
        height: 100%;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-box">
        <h1 style="margin-bottom:0.4rem;">Multimodal Product Classification</h1>
        <h3 style="margin-top:0; font-weight:500;">Rakuten Catalog Analysis and Modeling</h3>
        <p style="margin-top:1rem;">
            This project investigates multimodal product classification on the Rakuten catalog by combining
            <b>textual information</b> and <b>product images</b>. The overall workflow is structured into
            exploratory analysis, preprocessing, text modeling, image modeling, and multimodal learning.
        </p>
        <p class="small-note" style="margin-bottom:0;">
            Authors:  Artur Illenseer, Felix Bohn, Ion Chiorescu, and Sümeyra Özdemir
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("## Project Overview")

left_col, right_col = st.columns([1.15, 1], gap="large")

with left_col:
    st.markdown(
        """
        This application presents the full pipeline of the Rakuten product classification project.

        The main objective is to predict the target category <code>prdtypecode</code> for marketplace products
        by using:
        - product title (<code>designation</code>)
        - product description (<code>description</code>)
        - product image

        The project is organized into several stages:
        1. exploratory data analysis and preprocessing  
        2. text-only classification  
        3. image-only classification  
        4. multimodal learning  
        5. interactive prediction
        """,
        unsafe_allow_html=True
    )

with right_col:
    st.markdown(
        """
        <div class="card-box">
            <h4 style="margin-top:0;">Dataset Snapshot</h4>
            <p><b>Training samples:</b> 84,916</p>
            <p><b>Test samples:</b> 13,812</p>
            <p><b>Target classes:</b> 27</p>
            <p><b>Missing descriptions:</b> ~35.1%</p>
            <p><b>Modalities:</b> Text + Image</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("## Key Findings from Exploratory Analysis")

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown(
        """
        <div class="card-box">
            <h4 style="margin-top:0;">Text Signals</h4>
            <p>
                Product titles are the most reliable textual source because they are always present
                and usually contain concise category-defining keywords.
            </p>
            <p>
                Descriptions add complementary information, but they are noisier, more heterogeneous,
                and frequently missing.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="card-box">
            <h4 style="margin-top:0;">Category Distribution</h4>
            <p>
                The dataset contains 27 product categories with moderate class imbalance.
            </p>
            <p>
                Even though some classes are much larger than others, all categories still contain
                enough samples to support classification experiments.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="card-box">
            <h4 style="margin-top:0;">Preprocessing</h4>
            <p>
                Text cleaning included lowercasing, stopword removal, punctuation cleaning,
                whitespace normalization, and HTML cleanup.
            </p>
            <p>
                Repeated text blocks in descriptions were detected and removed to reduce noise.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("## Why This Project Matters")

st.markdown(
    """
    The Rakuten catalog is a strong example of a real-world multimodal classification task:

    - some categories can be recognized well from text alone  
    - some categories benefit strongly from visual information  
    - many difficult cases lie at the boundary between semantically similar product families  

    Because of this, the project does not stop at a single model type. Instead, it compares:
    - classical text models
    - transformer-based text models
    - image-only deep learning models
    - multimodal fusion approaches
    """
)

st.markdown("## Application Structure")

sections = [
    (
        "Data Exploration",
        "Dataset structure, class balance, text distributions, and category-specific token analysis."
    ),
    (
        "Text Modeling",
        "Classical baselines, advanced text representations, and transformer fine-tuning."
    ),
    (
        "Image Modeling",
        "CNN baselines, transfer learning with ResNet, and image-only classification experiments."
    ),
    (
        "Multimodal Modeling",
        "Fusion methods combining image and text features for stronger product classification."
    ),
    (
        "Predict",
        "Interactive inference page for testing image, text, and multimodal models."
    ),
]

cols = st.columns(len(sections), gap="medium")

for col, (title, description) in zip(cols, sections):
    with col:
        st.markdown(
            f"""
            <div class="card-box">
                <h4 style="margin-top:0;">{title}</h4>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("## Main Objectives")

obj1, obj2 = st.columns(2, gap="large")

with obj1:
    st.markdown(
        """
        - identify the most informative text sources for product classification  
        - understand the role of product descriptions and missing values  
        - evaluate the contribution of category-specific vocabulary  
        - compare sparse lexical methods with contextual transformer models  
        """
    )

with obj2:
    st.markdown(
        """
        - study the effect of image resolution, augmentation, and transfer learning  
        - compare image-only and text-only model families  
        - explore multimodal fusion to improve difficult category boundaries  
        - build a usable interactive prediction interface  
        """
    )

st.markdown("## Summary")

st.info(
    """
    The exploratory analysis shows that product titles provide the strongest and most stable textual signal,
    descriptions provide complementary but noisier information, and the dataset structure naturally motivates
    a multimodal approach. The following pages present how these observations were turned into text, image,
    and multimodal classification models.
    """
)