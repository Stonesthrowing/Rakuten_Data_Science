# %%
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

# %%
df = pd.merge(X_train, Y_train, on="Unnamed: 0")

print(df.shape)
df.head()

# %%
df_code = df[df['prdtypecode'] == 2582]
display(df_code.head(20))

from PIL import Image
import matplotlib.pyplot as plt
import os

for _, row in df_code.head(20).iterrows():
    filename = f"image_train/image_{row['imageid']}_product_{row['productid']}.jpg"
    path = os.path.join("images", filename)   # change folder if needed
    
    img = Image.open(path)
    
    plt.imshow(img)
    plt.title(filename)
    plt.axis("off")
    plt.show()

# %%
df["prdtypecode"].nunique()

# %%
df["prdtypecode"].value_counts().head(10)

# %%
df["prdtypecode"].describe()

# %%
import matplotlib.pyplot as plt

df["prdtypecode"].value_counts().head(20).plot(kind="bar")
plt.title("Top 20 most frequent product categories")
plt.show()

# %%
df.isnull().sum()

# %%
missing_desc = df["description"].isnull().mean()*100
print("Missing description (%):", missing_desc)

# %%
df["designation_length"] = df["designation"].str.len()
df["description_length"] = df["description"].str.len()

df[["designation_length","description_length"]].describe()

# %%
import matplotlib.pyplot as plt

df["designation_length"].hist(bins=50)
plt.title("Distribution of designation length")
plt.xlabel("Number of characters")
plt.ylabel("Frequency")
plt.show()

# %%

train_images = os.listdir("images/image_train")
test_images = os.listdir("images/image_test")

print("Number of training images:", len(train_images))
print("Number of test images:", len(test_images))

# %%
train_images[:10]

# %%
for i in range(5):
    row = df.iloc[i]
    image_name = f"image_{row.imageid}_product_{row.productid}.jpg"
    print(image_name in train_images)

# %%
correct = 0

for _, row in df.iterrows():
    image_name = f"image_{row.imageid}_product_{row.productid}.jpg"
    
    if image_name in train_images:
        correct += 1

print(f"{correct} out of {len(df)} correct")

# %% [markdown]
# # NEW FROM HERE

# %%
df["designation"].head(10)

# %%
df["designation"].sample(10, random_state=42)

# %%
df["designation"].str.len().describe()

# %%
df["designation"].isnull().sum()

# %%
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

# %%
df[["description","description_clean"]].sample(20, random_state=7)

# %%
# split each cleaned title into tokens
tokens = df["designation_clean"].str.split()
display(tokens.head())
# convert the list of tokens into a single column of words for an initial distribution count
all_words = tokens.explode()
# let see the first 30
all_words.value_counts().head(30)

# %%
import nltk
nltk.download('stopwords')

# %%
from nltk.corpus import stopwords
stop_words = set(stopwords.words(['french', 'english', 'german']))

# %%
filtered_words = all_words[~all_words.isin(stop_words)]
filtered_words.value_counts().head(30)

# %%
filtered_words = filtered_words[filtered_words.str.len() > 2]
filtered_words.value_counts().head(30)

# %%
print(f" there are {filtered_words.nunique()} unique words")
# statistics per words in designation
tokens_filtered = df["designation_clean"].str.split()
tokens_filtered.apply(len).describe()



