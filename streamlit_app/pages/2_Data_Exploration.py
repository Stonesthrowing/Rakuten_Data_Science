import streamlit as st

st.set_page_config(page_title="Data Exploration", layout="wide")

st.title("📊 Data Exploration")
st.markdown("### Exploratory Analysis of the Rakuten Product Dataset")

# --------------------------------------------------
# INTRODUCTION
# --------------------------------------------------
st.markdown("## Introduction")

st.markdown("""
This analysis investigates the structure of the Rakuten product dataset, focusing on textual features and category distribution.  
The goal is to understand product titles and descriptions, identify useful signals for classification, and analyze how these features relate to product categories.
""")

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------
st.markdown("## Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Training Samples", "84,916")

with col2:
    st.metric("Test Samples", "13,812")

with col3:
    st.metric("Number of Categories", "27")

st.markdown("""
Each product includes:
- **Text data**: title (`designation`) and description
- **Image data**
- **Target label**: `prdtypecode`
""")

# --------------------------------------------------
# VARIABLES TABLE
# --------------------------------------------------
st.markdown("## Dataset Structure")

st.table({
    "Variable": ["designation", "description", "productid", "imageid", "prdtypecode"],
    "Description": [
        "Product title",
        "Detailed product description (~35% missing)",
        "Unique product ID",
        "Image identifier",
        "Target category label"
    ]
})

# --------------------------------------------------
# TEXT INFORMATION
# --------------------------------------------------
st.markdown("## Textual Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**Product Titles (designation)**
- Always available
- Short and structured
- Highly informative for classification
""")

with col2:
    st.markdown("""
**Product Descriptions**
- ~35% missing
- Longer and more variable
- Provide additional context (attributes, materials, etc.)
""")

# --------------------------------------------------
# IMAGE DATA
# --------------------------------------------------
st.markdown("## Image Data")

st.markdown("""
Each product has an associated image.

**Naming convention:**
""")

# --------------------------------------------------
# CATEGORY DISTRIBUTION
# --------------------------------------------------
st.markdown("## Category Distribution")

st.markdown("""
The dataset is **moderately imbalanced**:
- Some categories have >10,000 samples
- Others have <1,000 samples
""")

st.info("⚠️ This is why **Macro F1 score** is used as the main evaluation metric.")

# 👉 buraya grafik koyacaksın (senin plotun)
st.markdown("📌 *Insert your category distribution plot here*")

# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------
st.markdown("## Text Preprocessing")

st.markdown("""
The preprocessing pipeline includes:
- Lowercasing
- Removing punctuation and special characters
- Removing stopwords (French, English, German)
- Removing short words (<3 characters)
- Removing HTML tags
- Normalizing whitespace
""")

st.markdown("""
🔢 **Important finding:**  
Numbers were **NOT removed**, because they may represent:
- product size
- version/model
- quantity

→ These are often useful for classification.
""")

# --------------------------------------------------
# TOKEN ANALYSIS
# --------------------------------------------------
st.markdown("## Token & Length Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**Titles**
- ~11.5 words on average
- Very consistent length
- Strong signal
""")

with col2:
    st.markdown("""
**Descriptions**
- ~84 words average
- Highly variable
- Sometimes missing
""")

st.markdown("📌 *Insert token distribution plots here*")

# --------------------------------------------------
# DEDUPLICATION
# --------------------------------------------------
st.markdown("## Description Deduplication")

st.markdown("""
Some descriptions contain repeated text blocks.

Solution:
- Detect repeated segments (>100 chars)
- Keep only first occurrence
""")

st.success("""
✔ Reduced noise  
✔ Lower extreme outliers  
✔ Preserved meaningful information
""")

# --------------------------------------------------
# VOCABULARY INSIGHTS
# --------------------------------------------------
st.markdown("## Vocabulary Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**Titles → Category Signals**
- piscine, kit, drone, sac
- directly indicate product type
""")

with col2:
    st.markdown("""
**Descriptions → Attributes**
- couleur, taille, qualité
- describe features, not category
""")

# --------------------------------------------------
# CATEGORY EXAMPLES
# --------------------------------------------------
st.markdown("## Category-Specific Patterns")

st.markdown("""
Examples:

- **2583 (Swimming pool)** → piscine, spa, pompe  
- **1560 (Furniture)** → meuble-related tokens  
- **1180 (Collectibles)** → figurines, masks  
""")

st.markdown("📌 *Insert category token plots here*")

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------
st.markdown("## Key Takeaways")

st.success("""
✔ Titles are the strongest signal for classification  
✔ Descriptions provide complementary context  
✔ Dataset is moderately imbalanced → use F1 metrics  
✔ Numeric tokens are informative  
✔ Deduplication improves data quality  
""")