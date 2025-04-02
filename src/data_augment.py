import pandas as pd
import unicodedata
import re
import string
from datasets import load_dataset
from sklearn.utils import shuffle
import random

# -----------------------------------------------
# TEXT CLEANING UTILS
# -----------------------------------------------

def normalize(text):
    return unicodedata.normalize("NFKC", text).lower().strip()

def is_too_punctuated(text, threshold=0.3):
    punct_ratio = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
    return punct_ratio <= threshold

def is_not_religious(text):
    filter_keywords = ["o prophet", "nay,", "verily", "lo!", "messenger", "punishment", "behold", "god", "allah"]
    return not any(kw in text for kw in filter_keywords)

def contains_letters(text):
    # True if text contains any Unicode letter (Latin, Arabic, Devanagari, etc.)
    return any(unicodedata.category(c).startswith('L') for c in text)

def is_clean(text):
    return contains_letters(text) and not bool(re.search(r"[�💖💁🏾👍🏾🤖📦🌍]", text))

def is_not_repetitive(text, max_repeat=5):
    tokens = text.split()
    most_common = max([tokens.count(tok) for tok in set(tokens)], default=0)
    return most_common <= max_repeat

# -----------------------------------------------
# MAIN PREPROCESS FUNCTION
# -----------------------------------------------
def preprocess_dataframe(df):
    df = df.copy()

    df["inputs"] = df["inputs"].astype(str).apply(normalize)
    df["targets"] = df["targets"].astype(str).apply(normalize)

    df = df[
        df["inputs"].notna() & df["targets"].notna() &
        (df["inputs"].str.strip().str.len() > 0) &
        (df["targets"].str.strip().str.len() > 0)
    ]

    df = df[
        df["inputs"].apply(is_clean) &
        df["targets"].apply(is_clean) &
        df["inputs"].apply(is_not_religious) &
        df["inputs"].apply(is_not_repetitive) &
        df["targets"].apply(is_not_repetitive) &
        df["targets"].str.split().str.len().ge(4) &
        (df["inputs"] != df["targets"]) &
        df["inputs"].apply(is_too_punctuated)
    ]

    df = df.drop_duplicates(subset=["inputs", "targets"])
    df = df[
        (df["inputs"].str.split().str.len() < 128) &
        (df["targets"].str.split().str.len() < 128)
    ]

    # ✅ Standardize instruction text
    df["instruction"] = "translate the following from english into hausa."

    return df.reset_index(drop=True)
