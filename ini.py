import pandas as pd
import numpy as np
import re
import nltk
# nltk.download('stopwords')  # this could be neccessary
from nltk.corpus import stopwords
stop_words = set(stopwords.words(['french', 'english', 'german']))

# removing repeating text blocks in description

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

def clean_txt_colmn(dtfrme, column):    # function to clean columns in a dataframe and generate a new column_clean 
    new = column + "_clean"    
    # replacing NaNs with empty strings
    dtfrme[new] = dtfrme[column].fillna("")    
    # remove html tags
    dtfrme[new] = dtfrme[new].str.replace(r"<.*?>", " ", regex=True)    
    # decode html entities &amp
    dtfrme[new] = dtfrme[new].str.replace(r"&\w+;", " ", regex=True)    
    # lowercase
    dtfrme[new] = dtfrme[new].str.lower()    
    # remove punctuation and special characters (parantheses ...)
    dtfrme[new] = dtfrme[new].str.replace(r"[^\w\s]", " ", regex=True)    
    # normalize spaces
    dtfrme[new] = dtfrme[new].str.replace(r"\s+", " ", regex=True)    
    # strip
    dtfrme[new] = dtfrme[new].str.strip()
    return dtfrme

def prepare_all_words(df, column, stop_words): # function to prepare vocabularies

    # tokenize
    tokens = df[column].str.split()

    # flatten into single column of words
    words = tokens.explode()

    # remove stopwords
    words = words[~words.isin(stop_words)]

    # remove short tokens
    words = words[words.str.len() > 2]

    return words



X_train = pd.read_csv("X_train_update.csv")
Y_train = pd.read_csv("Y_train_CVw08PX.csv")
X_test = pd.read_csv("X_test_update.csv")

df = pd.merge(X_train, Y_train, on="Unnamed: 0")

clean_txt_colmn(df,"designation")
clean_txt_colmn(df,"description")

design_words = prepare_all_words(df, "designation_clean", stop_words)
descrp_words = prepare_all_words(df, "description_clean", stop_words)

# remove reteated block of texts  only to descriptions
df["description_dedup"] = df["description_clean"].apply(remove_repeated_blocks)

df.to_csv("train_clean.csv", index=False)

