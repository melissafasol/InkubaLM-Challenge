import pandas as pd
import unicodedata
import re
import string
from datasets import load_dataset
from sklearn.utils import shuffle

# -----------------------------------------------
# TEXT CLEANING UTILS
# -----------------------------------------------
def normalize(text):
    return unicodedata.normalize("NFKC", text).lower().strip()

def is_clean(text):
    return not bool(re.search(r'[{}[\]@#^*_<>~]', text))

def is_too_punctuated(text, threshold=0.3):
    punct_ratio = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
    return punct_ratio <= threshold

def is_not_religious(text):
    filter_keywords = ["o prophet", "nay,", "verily", "lo!", "messenger", "punishment", "behold", "god", "allah"]
    return not any(kw in text for kw in filter_keywords)

# -----------------------------------------------
# SYNTHETIC BOOTSTRAP DATA
# -----------------------------------------------
def create_bootstrap_mt_examples(language: str, level: str = "simple") -> pd.DataFrame:
    examples = {
        "hausa": {
            "simple": [
                ["Hello, how are you?", "Sannu, ya kake?"],
                ["Thank you very much!", "Na gode sosai!"],
                ["What is your name?", "Menene sunanka?"]
            ],
            "medium": [
                ["The weather today is quite pleasant.", "Yanayin yau yana da daɗi sosai."],
                ["I am learning how to speak Hausa fluently.", "Ina koyon yadda ake magana da Hausa sosai."],
                ["Can you help me find the nearest bus stop?", "Za ka iya taimaka mini in sami tasha mafi kusa?"]
            ],
            "rich": [
                ["Despite the challenges, we managed to finish the project on time.",
                 "Duk da ƙalubale, mun gama aikin a kan lokaci."],
                ["Language learning is a journey that requires patience and consistency.",
                 "Koyon harshe tafiya ce da ke bukatar haƙuri da jajircewa."],
                ["The conference focused on ways to improve education in rural communities.",
                 "Taron ya mayar da hankali kan hanyoyin inganta ilimi a yankunan karkara."]
            ]
        }
    }
    
    level_data = examples[language.lower()][level]
    return pd.DataFrame({
        "ID": [f"bootstrap_{language}_{level}_{i}" for i in range(len(level_data))],
        "task": ["mmt"] * len(level_data),
        "langs": ["eng-" + language[:3]] * len(level_data),
        "data_source": ["synthetic"] * len(level_data),
        "instruction": [f"translate the following from English into {language.lower()}." for _ in level_data],
        "inputs": [pair[0].lower() for pair in level_data],
        "targets": [pair[1].lower() for pair in level_data]
    })

# -----------------------------------------------
# OPUS100 CLEANING + FORMATTING
# -----------------------------------------------
def load_and_clean_opus(language="hausa", limit=1000):
    print("📥 Loading OPUS dataset...")
    ds = load_dataset("opus100", "en-ha")  # will auto-download if not cached
    subset = ds["train"].select(range(limit))
    
    df = pd.DataFrame({
        "inputs": [ex["translation"]["en"] for ex in subset],
        "targets": [ex["translation"]["ha"] for ex in subset]
    })

    # Normalize and clean
    df["inputs"] = df["inputs"].apply(normalize)
    df["targets"] = df["targets"].apply(normalize)

    df = df[
        df["inputs"].notna() & df["targets"].notna() &
        (df["inputs"].str.strip().str.len() > 0) &
        (df["targets"].str.strip().str.len() > 0)
    ]

    df = df[
        df["inputs"].apply(is_clean) &
        df["targets"].apply(is_clean) &
        df["inputs"].apply(is_not_religious) &
        df["targets"].str.split().str.len().ge(4) &
        (df["inputs"] != df["targets"]) &
        df["inputs"].apply(is_too_punctuated)
    ]

    df = df.drop_duplicates(subset=["inputs", "targets"])
    df = df[
        (df["inputs"].str.split().str.len() < 128) &
        (df["targets"].str.split().str.len() < 128)
    ]

    return pd.DataFrame({
        "ID": [f"opus_{language}_{i}" for i in range(len(df))],
        "task": ["mmt"] * len(df),
        "langs": ["eng-" + language[:3]] * len(df),
        "data_source": ["opus100"] * len(df),
        "instruction": [f"translate the following from English into {language.lower()}." for _ in range(len(df))],
        "inputs": df["inputs"].tolist(),
        "targets": df["targets"].tolist()
    })

# -----------------------------------------------
# MAIN PIPELINE FUNCTION
# -----------------------------------------------
def build_hausa_mt_dataset():
    df_opus = load_and_clean_opus()

    bootstrap_df = pd.concat([
        create_bootstrap_mt_examples("hausa", "simple"),
        create_bootstrap_mt_examples("hausa", "medium"),
        create_bootstrap_mt_examples("hausa", "rich")
    ], ignore_index=True)

    final_df = pd.concat([df_opus, bootstrap_df], ignore_index=True)
    final_df = shuffle(final_df, random_state=42).reset_index(drop=True)

    return final_df
