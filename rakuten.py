#!/usr/bin/env python
# coding: utf-8

# In[26]:


get_ipython().system('uv pip install nltk')


# In[1]:


import os
os.chdir(path=r"C:\Users\Ion\Documents\working now\job\datascientist\rakuten_project")
os.getcwd()


# In[2]:


display(os.listdir())
display(os.listdir("images"))
display(os.listdir("images/image_train")[:10])


# In[2]:


import pandas as pd

X_train = pd.read_csv("X_train_update.csv")
Y_train = pd.read_csv("Y_train_CVw08PX.csv")
X_test = pd.read_csv("X_test_update.csv")

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)

display(X_train.head())
display(Y_train.head())

display(X_train.info())
display(X_train.describe(include="all"))


# In[3]:


df = pd.merge(X_train, Y_train, on="Unnamed: 0")

print(df.shape)
df.head()


# In[21]:


df_code = df[df['prdtypecode'] == 1180]
display(df_code.head(10))
with pd.option_context("display.max_colwidth", None):
    print(df_code.designation.head(10))

from PIL import Image
import matplotlib.pyplot as plt
import os

for _, row in df_code.head(10).iterrows():
    filename = f"image_train/image_{row['imageid']}_product_{row['productid']}.jpg"
    path = os.path.join("images", filename)   # change folder if needed

    img = Image.open(path)

    plt.imshow(img)
    plt.title(filename)
    plt.axis("off")
    plt.show()


# In[5]:


df["prdtypecode"].nunique()


# In[8]:


df["prdtypecode"].value_counts().head(10)


# In[9]:


df["prdtypecode"].describe()


# In[8]:


import matplotlib.pyplot as plt

df["prdtypecode"].value_counts().head(20).plot(kind="bar")
plt.title("Top 20 most frequent product categories")
plt.show()


# In[10]:


# least represented categories:

df["prdtypecode"].value_counts().tail(7).sort_values().plot(kind="bar")
plt.title("Top 7 least frequent product categories")
plt.show()


# In[12]:


df.isnull().sum()


# In[13]:


missing_desc = df["description"].isnull().mean()*100
print("Missing description (%):", missing_desc)


# In[14]:


df["designation_length"] = df["designation"].str.len()
df["description_length"] = df["description"].str.len()

df[["designation_length","description_length"]].describe()


# In[15]:


import matplotlib.pyplot as plt

df["designation_length"].hist(bins=50)
plt.title("Distribution of designation length")
plt.xlabel("Number of characters")
plt.ylabel("Frequency")
plt.show()


# In[41]:


train_images = os.listdir("images/image_train")
test_images = os.listdir("images/image_test")

print("Number of training images:", len(train_images))
print("Number of test images:", len(test_images))


# In[17]:


train_images[:10]


# In[19]:


for i in range(5):
    row = df.iloc[i]
    image_name = f"image_{row.imageid}_product_{row.productid}.jpg"
    print(image_name in train_images)


# In[43]:


correct = 0

for _, row in df.iterrows():
    image_name = f"image_{row.imageid}_product_{row.productid}.jpg"

    if image_name in train_images:
        correct += 1

print(f"{correct} out of {len(df)} correct")


# In[8]:


df["designation"].head(10)


# In[9]:


df["designation"].sample(10, random_state=42)


# In[10]:


df["designation"].str.len().describe()


# In[11]:


df["designation"].isnull().sum()


# In[49]:


def clean_txt_colmn(dtfrme, column):    
    new = column + "_clean"    
    # replacing NaNs with empty strings
    dtfrme[new] = dtfrme[column].fillna("")    
    # remove html tags
    dtfrme[new] = dtfrme[new].str.replace(r"<.*?>", " ", regex=True)    
    # decode html entities &amp
    dtfrme[new] = dtfrme[new].str.replace(r"&\w+;", " ", regex=True)    
    # lowercase
    dtfrme[new] = dtfrme[new].str.lower()    
    # remove punctuation and special characters la parantheses fe
    dtfrme[new] = dtfrme[new].str.replace(r"[^\w\s]", " ", regex=True)    
    # normalize spaces
    dtfrme[new] = dtfrme[new].str.replace(r"\s+", " ", regex=True)    
    # strip
    dtfrme[new] = dtfrme[new].str.strip()
    return dtfrme


clean_txt_colmn(df,"designation")
clean_txt_colmn(df,"description")


# In[51]:


# statistics on characters

df["designation_clean_length"] = df["designation_clean"].str.len()
df["description_clean_length"] = df["description_clean"].str.len()

df[["designation_clean_length","description_clean_length"]].describe()


# In[24]:


df.loc[df["description_clean_length"].idxmax(), "description_clean"]
df.loc[df["description_clean_length"].idxmax()]


# In[32]:


df.loc[64317, "description_clean"]


# In[ ]:


# the image which has a description of 12K characters

from PIL import Image
import matplotlib.pyplot as plt
import os


filename = f"image_train/image_1277618412_product_4023550194.jpg"
path = os.path.join("images", filename)   # change folder if needed

img = Image.open(path)

plt.imshow(img)
plt.title(filename)
plt.axis("off")
plt.show()


# In[26]:


# top 10 longest descriptions
df.nlargest(10, "description_clean_length")[["Unnamed: 0", "productid", "imageid", "description_clean_length"]]


# In[52]:


# statistics on tokens

df["designation_clean_length"] = df["designation_clean"].str.split().str.len()
df["description_clean_length"] = df["description_clean"].str.split().str.len()

df[["designation_clean_length","description_clean_length"]].describe()


# In[53]:


df["designation_clean_length"] = df["designation_clean"].str.split().str.len()
df["description_clean_length"] = df["description_clean"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(12,5))

df["designation_clean_length"].hist(bins=50, ax=axes[0])
axes[0].set_title("Distribution of designation_clean token count")
axes[0].set_xlabel("Number of tokens")
axes[0].set_ylabel("Frequency")

df["description_clean_length"].hist(bins=50, ax=axes[1])
axes[1].set_title("Distribution of description_clean token count")
axes[1].set_xlabel("Number of tokens")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()



# In[54]:


import numpy as np

# lengths
df["designation_clean_length"] = df["designation_clean"].str.split().str.len()
desc_nonempty = df.loc[df["description_clean"].str.strip() != "", "description_clean"]
desc_nonempty_len = desc_nonempty.str.split().str.len()

def outlier_ratio(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((series < lower) | (series > upper)).sum()
    return outliers / len(series)

ratio_design = outlier_ratio(df["designation_clean_length"])
ratio_descr  = outlier_ratio(desc_nonempty_len)

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# designation
df["designation_clean_length"].plot.box(ax=axes[0])
axes[0].set_title("designation_clean token count")
axes[0].set_ylabel("Number of tokens")
axes[0].text(
    0.65, 0.92,                      # inside the plot area (right/top)
    f"Outliers: {ratio_design:.2%}",
    transform=axes[0].transAxes,
    ha="left", va="center",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
)

# description (non-empty only)
desc_nonempty_len.plot.box(ax=axes[1])
axes[1].set_title("description_clean token count\n(non-empty only)")
axes[1].set_ylabel("Number of tokens")
axes[1].text(
    0.65, 0.92,
    f"Outliers: {ratio_descr:.2%}",
    transform=axes[1].transAxes,
    ha="left", va="center",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
)

plt.tight_layout()
plt.show()


# In[ ]:


# search for copied block of text in textual columns

import re

def has_repeated_block(text, min_block_len=100):
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < min_block_len * 2:
        return False
    # look for a block repeated at least twice
    pattern = rf"(.{{{min_block_len},}})\1+"
    return re.search(pattern, text) is not None

# detect rows with repeated blocks
df["descr_dup_block"] = df["description_clean"].apply(has_repeated_block)
df["desig_dup_block"] = df["designation_clean"].apply(has_repeated_block)

# summary
print("Rows with duplicated blocks in description:", df["descr_dup_block"].sum())
print("Percentage affected:", round(100 * df["descr_dup_block"].mean(), 2), "%")
print("\n\nRows with duplicated blocks in designation:", df["desig_dup_block"].sum())
print("Percentage affected:", round(100 * df["desig_dup_block"].mean(), 2), "%")



# In[37]:


# inspect a few examples

import re

def find_repeated_blocks(text, min_block_len=80):
    """
    Detect consecutive repeated blocks in a text.
    Returns: (block, repetitions, block_length)
    """
    if not isinstance(text, str):
        return []

    pattern = re.compile(r"(.{" + str(min_block_len) + r",}?)(\1+)", re.DOTALL)

    matches = []

    for m in pattern.finditer(text):
        block = m.group(1)
        repeated = m.group(2)

        repetitions = 1 + len(repeated) // len(block)

        matches.append((block, repetitions, len(block)))

    return matches


top5 = df.nlargest(5, "description_clean_length")[[
    "Unnamed: 0",
    "description_clean_length",
    "description_clean"
]]

for _, row in top5.iterrows():

    print("="*100)
    print("ID:", row["Unnamed: 0"])
    print("Description length:", row["description_clean_length"])

    repeated_blocks = find_repeated_blocks(row["description_clean"], 80)

    if not repeated_blocks:
        print("No repeated blocks found.")
        continue

    print("Repeated blocks found:", len(repeated_blocks))

    for i, (block, reps, block_len) in enumerate(repeated_blocks[:5], 1):

        print("-"*80)
        print("Block", i)
        print("Block length:", block_len)
        print("Repetitions:", reps)
        print("Preview:")
        print(block[:300])

        if len(block) > 300:
            print("...")


# In[55]:


# removing repeating text blocks in description

import re

def remove_repeated_blocks(text, min_block_len=100):
    """
    Remove consecutively repeated text blocks of length >= min_block_len.
    Keeps the first occurrence only.
    """
    if not isinstance(text, str):
        return text

    text = text.strip()
    if len(text) < 2 * min_block_len:
        return text

    # build regex safely
    pattern = re.compile(r"(.{" + str(min_block_len) + r",}?)(?:\1)+", re.DOTALL)

    previous = None
    while text != previous:
        previous = text

        # collapse repeated consecutive blocks
        text = pattern.sub(r"\1", text)

        # normalize spaces created during replacement
        text = re.sub(r"\s+", " ", text).strip()

    return text


# apply only to descriptions
df["description_dedup"] = df["description_clean"].apply(remove_repeated_blocks)

# compare before vs after
df["description_dedup_length_chars"] = df["description_dedup"].str.len()
df["description_dedup_length_words"] = df["description_dedup"].str.split().str.len()

df["removed_chars"] = df["description_clean"].str.len() - df["description_dedup_length_chars"]
df["removed_words"] = df["description_clean"].str.split().str.len() - df["description_dedup_length_words"]

# summary
affected_after_cleaning = (df["removed_chars"] > 0).sum()

print("Rows changed after deduplication:", affected_after_cleaning)
print("Percentage changed:", round(100 * affected_after_cleaning / len(df), 2), "%")

display(
    df.loc[df["removed_chars"] > 0,["Unnamed: 0", "description_clean_length", "description_dedup_length_words", "removed_words", "removed_chars"]].sort_values("removed_chars", ascending=False).head(10))


# In[56]:


# token lengths
df["description_clean_length"] = df["description_clean"].str.split().str.len()
df["description_dedup_length"] = df["description_dedup"].str.split().str.len()

# exclude empty descriptions
desc_clean = df.loc[df["description_clean"].str.strip() != "", "description_clean_length"]
desc_dedup = df.loc[df["description_dedup"].str.strip() != "", "description_dedup_length"]

def box_stats(series):
    q1 = series.quantile(0.25)
    q2 = series.quantile(0.50)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = max(series.min(), q1 - 1.5 * iqr)
    upper = min(series.max(), q3 + 1.5 * iqr)
    outliers = ((series < lower) | (series > upper)).sum()
    ratio = outliers / len(series)
    return q1, q2, q3, lower, upper, ratio

stats_clean = box_stats(desc_clean)
stats_dedup = box_stats(desc_dedup)

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# description_clean
desc_clean.plot.box(ax=axes[0])
axes[0].set_title("description_clean token count (non-empty)")
axes[0].set_ylabel("Number of tokens")

text_clean = (
    f"Q1: {stats_clean[0]:.0f}\n"
    f"Median: {stats_clean[1]:.0f}\n"
    f"Q3: {stats_clean[2]:.0f}\n"
    f"Lower whisker: {stats_clean[3]:.0f}\n"
    f"Upper whisker: {stats_clean[4]:.0f}\n"
    f"Outliers: {stats_clean[5]:.2%}"
)

axes[0].text(
    0.65, 0.6,
    text_clean,
    transform=axes[0].transAxes,
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
)

# description_dedup
desc_dedup.plot.box(ax=axes[1])
axes[1].set_title("description_dedup token count (non-empty)")
axes[1].set_ylabel("Number of tokens")

text_dedup = (
    f"Q1: {stats_dedup[0]:.0f}\n"
    f"Median: {stats_dedup[1]:.0f}\n"
    f"Q3: {stats_dedup[2]:.0f}\n"
    f"Lower whisker: {stats_dedup[3]:.0f}\n"
    f"Upper whisker: {stats_dedup[4]:.0f}\n"
    f"Outliers: {stats_dedup[5]:.2%}"
)

axes[1].text(
    0.65, 0.6,
    text_dedup,
    transform=axes[1].transAxes,
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
)

plt.tight_layout()
plt.show()


# In[20]:


df[["description","description_clean"]].sample(20, random_state=7)


# In[5]:


# split each cleaned title into tokens
tokens = df["designation_clean"].str.split()
display(tokens.head())
# convert the list of tokens into a single column of words for an initial distribution count
all_words = tokens.explode()
# let see the first 30
all_words.value_counts().head(30)


# In[17]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words(['french', 'english', 'german']))


# In[18]:


def prepare_all_words(df, column, stop_words):

    # tokenize
    tokens = df[column].str.split()

    # flatten into single column of words
    words = tokens.explode()

    # remove stopwords
    words = words[~words.isin(stop_words)]

    # remove short tokens
    words = words[words.str.len() > 2]

    return words

design_words = prepare_all_words(df, "designation_clean", stop_words)
descrp_words = prepare_all_words(df, "description_clean", stop_words)

display(len(design_words.unique()))
display(len(descrp_words.unique()))


# In[19]:


display(design_words.value_counts().head(10))
display(descrp_words.value_counts().head(10))


# In[12]:


df[df["designation_clean"].str.contains(r"\d", regex=True)][["designation_clean","description_clean"]].head(10)


# In[32]:


print(f" there are {filtered_words.nunique()} unique words")
# statistics per words in designation
tokens_filtered = df["designation_clean"].str.split()
tokens_filtered.apply(len).describe()


# In[11]:


top_categories = df["prdtypecode"].value_counts().head(10).index        # get the 10 most frequent categories
df["tokens"] = df["designation_clean"].str.split()                      # create token lists from titles
df_tokens = df[["prdtypecode","tokens"]].explode("tokens")              # explode tokens so each word becomes a row
df_tokens = df_tokens[~df_tokens["tokens"].isin(stop_words)]            # keep only filtered tokens (no stopwords, length > 2)
df_tokens = df_tokens[df_tokens["tokens"].str.len() > 2]            
df_tokens = df_tokens[df_tokens["prdtypecode"].isin(top_categories)]    # keep only the top 10 categories
                                                                        # compute token frequency per category
token_counts = (df_tokens.groupby(["prdtypecode","tokens"]).size().reset_index(name="count"))

                                                                        # select top 5 tokens per category
top_tokens = (token_counts.sort_values(["prdtypecode","count"], ascending=[True, False]).groupby("prdtypecode").head(5))


for cat in top_categories:    
    data = top_tokens[top_tokens["prdtypecode"] == cat]    
    data.plot(
        x="tokens",
        y="count",
        kind="bar",
        legend=False
    )

    plt.title(f"Top tokens for category {cat}")
    plt.ylabel("frequency")
    plt.xlabel("token")
    plt.show();


# In[12]:


# for category 1180
cat_df = df[df["prdtypecode"] == 1180]
tokens = cat_df["designation_clean"].str.split()
words = tokens.explode()
words = words[~words.isin(stop_words)]
words = words[words.str.len() > 2]
top_words = words.value_counts().head(5)
top_words.plot(kind="bar")

plt.title("Top 5 tokens in designation for category 1180")
plt.ylabel("frequency")
plt.xlabel("token")

plt.show()


# In[63]:


# for temporary use 
df_raw_260312 = df.copy()
df_raw_260312.head(10)
df.to_csv("df_raw_260312.csv", index=False)


# In[ ]:


df["description_clean_ini"] = df["description_clean"]
df.head(5)


# In[67]:


df["description_clean"] = df["description_dedup"]
df.head(5)


# In[70]:


# tokenize
design_tokens = df["designation_clean"].str.split().explode()
descr_tokens = df["description_dedup"].str.split().explode()

# remove empty tokens if any
design_tokens = design_tokens.dropna()
descr_tokens = descr_tokens.dropna()

# detect numeric tokens
design_numeric = design_tokens.str.fullmatch(r"\d+")
descr_numeric = descr_tokens.str.fullmatch(r"\d+")

# counts
design_total = len(design_tokens)
descr_total = len(descr_tokens)

design_numeric_count = design_numeric.sum()
descr_numeric_count = descr_numeric.sum()

# percentages
design_numeric_pct = 100 * design_numeric_count / design_total
descr_numeric_pct = 100 * descr_numeric_count / descr_total

print("Designation numeric tokens:", design_numeric_count)
print("Designation total tokens:", design_total)
print("Designation percentage numeric:", round(design_numeric_pct,2), "%")

print()

print("Description numeric tokens:", descr_numeric_count)
print("Description total tokens:", descr_total)
print("Description percentage numeric:", round(descr_numeric_pct,2), "%")


# In[69]:


def top_word_in_caregory(df, columns, topN=5, top_categ=1, category=None):
    """
    Display side-by-side bar charts of the most frequent tokens for a selected
    product category, using one or more cleaned text columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least:
        - 'prdtypecode'
        - the cleaned text columns '<column>_clean'

    columns : list of str
        List of original column names, without the suffix '_clean'.
        Example:
        ["designation", "description"]

    topN : int, default=5
        Number of most frequent tokens to display in each chart.

    top_categ : int, default=1
        Rank of the category by frequency.
        Used only if 'category' is not provided.
        Example:
        1 = most frequent category
        2 = second most frequent category

    category : int or None, default=None
        Specific prdtypecode to analyze.
        If provided, this overrides 'top_categ'.

    Behavior
    --------
    The function:
    1. Selects a category either by its rank in frequency or by an explicit code.
    2. Filters the dataframe to rows belonging to that category.
    3. Uses the corresponding cleaned columns '<column>_clean'.
    4. Tokenizes the text.
    5. Removes stopwords, short tokens (length <= 2), and purely numeric tokens.
    6. Computes the topN most frequent tokens.
    7. Displays one bar chart per column, side by side.

    Notes
    -----
    - Cleaned columns such as 'designation_clean' and 'description_clean'
      must already exist before using this function.
    - If a cleaned column is missing, a warning is displayed.
    - The variable 'stop_words' must already be defined.
    """

    category_counts = df["prdtypecode"].value_counts()

    if category is not None:
        chosen_category = category
    else:
        chosen_category = category_counts.index[top_categ - 1]

    if chosen_category not in category_counts.index:
        print(f"Warning: category {chosen_category} not found in 'prdtypecode'.")
        return

    cat_df = df[df["prdtypecode"] == chosen_category]

    print(f"Selected category: {chosen_category}")
    print(f"Number of samples: {category_counts[chosen_category]}")

    fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 4))

    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        clean_col = col + "_clean"

        if clean_col not in df.columns:
            print(f"Warning: column '{clean_col}' not found. Prepare it beforehand.")
            ax.set_visible(False)
            continue

        words = cat_df[clean_col].str.split().explode()
        words = words[~words.isin(stop_words)]
        words = words[words.str.len() > 2]
        words = words[~words.str.isnumeric()]

        top_words = words.value_counts().head(topN)

        top_words.plot(kind="bar", ax=ax)

        ax.set_title(col, fontsize=16)
        ax.set_ylabel("frequency", fontsize=14)
        ax.set_xlabel("token", fontsize=14)

        ax.tick_params(axis="x", labelrotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    plt.show()



top_word_in_caregory(df, ["designation", "description"], topN=7)
top_word_in_caregory(df, ["designation", "description"], topN=7, top_categ=2)
top_word_in_caregory(df, ["designation", "description"], topN=7, category=1180)

