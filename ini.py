from pathlib import Path
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


stop_words = set().union(
    stopwords.words("french"),
    stopwords.words("english"),
    stopwords.words("german")
)

def remove_repeated_blocks(text, min_block_len=100, max_text_len=2000):
    if not isinstance(text, str):
        return text

    text = text.strip()

    if len(text) < 2 * min_block_len:
        return text

    if len(text) > max_text_len:
        return text

    pattern = re.compile(r"(.{" + str(min_block_len) + r",}?)(?:\1)+", re.DOTALL)

    previous = None
    while text != previous:
        previous = text
        text = pattern.sub(r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()

    return text

def clean_txt_colmn(dtfrme, column):
    new = column + "_clean"
    dtfrme[new] = dtfrme[column].fillna("")
    dtfrme[new] = dtfrme[new].str.replace(r"<.*?>", " ", regex=True)
    dtfrme[new] = dtfrme[new].str.replace(r"&\w+;", " ", regex=True)
    dtfrme[new] = dtfrme[new].str.lower()
    dtfrme[new] = dtfrme[new].str.replace(r"[^\w\s]", " ", regex=True)
    dtfrme[new] = dtfrme[new].str.replace(r"\s+", " ", regex=True)
    dtfrme[new] = dtfrme[new].str.strip()
    return dtfrme

def prepare_all_words(df, column, stop_words):
    tokens = df[column].str.split()
    words = tokens.explode()
    words = words[~words.isin(stop_words)]
    words = words[words.str.len() > 2]
    return words

PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data"

X_train = pd.read_csv(DATA_PATH / "X_train_update.csv").head(50)
Y_train = pd.read_csv(DATA_PATH / "Y_train_CVw08PX.csv").head(50)
X_test = pd.read_csv(DATA_PATH / "X_test_update.csv").head(50)

df = pd.merge(X_train, Y_train, on="Unnamed: 0")

df = df.dropna()

df = clean_txt_colmn(df, "designation")
df = clean_txt_colmn(df, "description")

design_words = prepare_all_words(df, "designation_clean", stop_words)
descrp_words = prepare_all_words(df, "description_clean", stop_words)

df["description_dedup"] = df["description_clean"]
mask = df["description_clean"].str.len().between(200, 2000)
df.loc[mask, "description_dedup"] = df.loc[mask, "description_clean"].apply(remove_repeated_blocks)

print("finished")
print(df.head())
print(df[["designation", "designation_clean"]].head())
print(df.shape)