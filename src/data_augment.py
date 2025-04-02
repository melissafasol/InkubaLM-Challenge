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
# UNIVERSAL PREPROCESSING FUNCTION
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
        df["targets"].str.split().str.len().ge(4) &
        (df["inputs"] != df["targets"]) &
        df["inputs"].apply(is_too_punctuated)
    ]

    df = df.drop_duplicates(subset=["inputs", "targets"])
    df = df[
        (df["inputs"].str.split().str.len() < 128) &
        (df["targets"].str.split().str.len() < 128)
    ]

    return df.reset_index(drop=True)

# -----------------------------------------------
# SYNTHETIC BOOTSTRAP DATA
# -----------------------------------------------
def create_bootstrap_mt_examples(language: str) -> pd.DataFrame:
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

    rows = []
    for level, pairs in examples[language.lower()].items():
        for i, (src, tgt) in enumerate(pairs):
            rows.append({
                "ID": f"bootstrap_{language}_{level}_{i}",
                "task": "mmt",
                "langs": f"eng-{language[:3]}",
                "data_source": "synthetic",
                "instruction": f"translate the following from English into {language.lower()}.",
                "inputs": src.lower(),
                "targets": tgt.lower()
            })

    return pd.DataFrame(rows)

# -----------------------------------------------
# OPUS100 LOADING + CLEANING
# -----------------------------------------------
def load_and_clean_opus(language="hausa", limit=1000):
    print("📥 Loading OPUS dataset...")
    ds = load_dataset("opus100", "en-ha")
    subset = ds["train"].select(range(limit))

    df = pd.DataFrame({
        "inputs": [ex["translation"]["en"] for ex in subset],
        "targets": [ex["translation"]["ha"] for ex in subset]
    })

    df_cleaned = preprocess_dataframe(df)

    df_cleaned["ID"] = [f"opus_{language}_{i}" for i in range(len(df_cleaned))]
    df_cleaned["task"] = "mmt"
    df_cleaned["langs"] = f"eng-{language[:3]}"
    df_cleaned["data_source"] = "opus100"
    df_cleaned["instruction"] = f"translate the following from English into {language.lower()}."

    return df_cleaned

# -----------------------------------------------
# MAIN PIPELINE FUNCTION
# -----------------------------------------------
def build_hausa_mt_dataset(custom_df=None):
    df_opus = load_and_clean_opus()

    df_bootstrap = create_bootstrap_mt_examples("hausa")

    if custom_df is not None:
        df_custom_clean = preprocess_dataframe(custom_df)
        df_custom_clean["ID"] = [f"custom_hausa_{i}" for i in range(len(df_custom_clean))]
        df_custom_clean["task"] = "mmt"
        df_custom_clean["langs"] = "eng-hau"
        df_custom_clean["data_source"] = "custom"
        df_custom_clean["instruction"] = "translate the following from English into hausa."
        all_data = pd.concat([df_opus, df_bootstrap, df_custom_clean], ignore_index=True)
    else:
        all_data = pd.concat([df_opus, df_bootstrap], ignore_index=True)

    return shuffle(all_data, random_state=42).reset_index(drop=True)
