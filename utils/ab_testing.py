import os
import pandas as pd
import numpy as np
from typing import List, Dict
from datasets import Dataset, DatasetDict


def process_likelihood(likelihood_str: str) -> List[float]:
    clean_str = (
        likelihood_str.replace("tensor(", "").replace(")", "").strip()
        .replace("[[", "").replace("]]", "").strip()
        .replace(" device='cuda:0'", "").replace(" dtype=torch.float16", "").strip()
        .replace("tensor", "").strip()
        .replace(",,", ",")
    )
    return [float(x) for x in clean_str.split(",") if x.strip()]


def load_and_normalize_parquet(path: str, lang_col: str = 'langs') -> pd.DataFrame:
    df = pd.read_parquet(path)
    df[lang_col] = df[lang_col].str.lower().str.strip()
    return df


def filter_languages(df: pd.DataFrame, lang_filters: Dict[str, str]) -> DatasetDict:
    return DatasetDict({
        lang: Dataset.from_pandas(df[df['langs'].str.contains(pattern)])
        for lang, pattern in lang_filters.items()
    })


def create_submission(output_path: str, test_flag: bool) -> pd.DataFrame:
    def load_file(filename):
        full_path = os.path.join(output_path, filename)
        print(f"Loading: {full_path}")
        return pd.read_csv(full_path)

    try:
        suffix = "_dev.csv" if test_flag else ".csv"
        sent_hausa = load_file(f"hau_sent_prediction{suffix}")
        sent_swahili = load_file(f"swa_sent_prediction{suffix}")
        mt_hausa = load_file(f"hau_mt_prediction{suffix}")
        mt_swahili = load_file(f"swa_mt_prediction{suffix}")
        xnli_hausa = load_file(f"hau_xnli_prediction{suffix}")
        xnli_swahili = load_file(f"swa_xnli_prediction{suffix}")
        filename = "submission_test.csv" if test_flag else "submission.csv"
    except FileNotFoundError as e:
        print("One or more prediction files are missing. Please complete all tasks before submission.")
        raise e

    def process_classification(df):
        df = df.copy()
        if "Log-Likelihood" in df.columns:
            df["Response"] = df["Log-Likelihood"]
        df["Response"] = df["Response"].apply(process_likelihood).apply(np.argmax)
        return df[["ID", "Response", "Targets"]] if test_flag else df[["ID", "Response"]]

    def process_mt(df):
        print("Preview MT df:")
        print(df.head(3))
        try:
            return df[["ID", "Response", "Targets"]] if test_flag else df[["ID", "Response"]]
        except KeyError as e:
            print("Error: One or more columns missing in MT prediction file.")
            raise e

    sent_df = pd.concat([process_classification(sent_hausa), process_classification(sent_swahili)], ignore_index=True)
    xnli_df = pd.concat([process_classification(xnli_hausa), process_classification(xnli_swahili)], ignore_index=True)
    mt_df = pd.concat([process_mt(mt_hausa), process_mt(mt_swahili)], ignore_index=True)

    submission = pd.concat([sent_df, xnli_df, mt_df], ignore_index=True)
    full_path = os.path.join(output_path, filename)
    submission.to_csv(full_path, index=False)
    print(f"✅ Submission saved to: {full_path}")
    return submission


def run_inference_on_tasks(
    output_path: str,
    lora_model,
    tokenizer,
    new_model_function,
    sample_size: int = 50,
    instruction_map: dict = None
):
    BASE_PROMPT = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction: {}\n\n"
        "### Response:"
    )

    default_instruction_map = {
        "sent": "Please identify the sentiment reflected in this text based on the following guidelines: Positive: if a text implies positive sentiment, attitude, and emotional state. Negative: if a text implies negative sentiment or emotion. Neutral: if a text does not imply positive or negative language directly or indirectly. Provide sentiment labels only",
        "xnli": "Is the following question True, False or Neither?",
        "mt": "Translate the following from {input_lang} into {output_lang}."
    }

    instructions = instruction_map or default_instruction_map

    parquet_paths = {
        "sent": "hf://datasets/lelapa/SentimentTrain/data/train-00000-of-00001.parquet",
        "xnli": "hf://datasets/lelapa/XNLITrain/data/train-00000-of-00001.parquet",
        "mt": "hf://datasets/lelapa/MTTrain/data/train-00000-of-00001.parquet",
    }

    lang_filters = {
        "swahili": "swa",
        "hausa": "hau"
    }

    datasets = {}
    for task, path in parquet_paths.items():
        df = load_and_normalize_parquet(path)
        if task == "mt":
            dataset_dict = filter_languages(df, {
                "swahili": "eng-swa",
                "hausa": "eng-hau"
            })
            for lang, ds in dataset_dict.items():
                if ds.num_rows == 0:
                    print(f"Warning: {lang.capitalize()} MT dataset is empty.")
        else:
            dataset_dict = filter_languages(df, lang_filters)
        datasets[task] = dataset_dict

    for task_key, dataset_dict in datasets.items():
        for lang, dataset in dataset_dict.items():
            print(f"\n🔁 Running task: {task_key} | Language: {lang}")
            filename = f"{lang[:3]}_{task_key}_prediction_dev.csv"
            filepath = os.path.join(output_path, filename)

            instruction = (
                instructions[task_key].format(input_lang=lang, output_lang="english")
                if task_key == "mt" else instructions[task_key]
            )

            new_model_function.main(
                model=lora_model,
                tokenizer=tokenizer,
                BASE_PROMPT=BASE_PROMPT,
                task_instruction=instruction,
                dataset=dataset,
                csv_file_path=filepath,
                custom_instruct=True,
                sample_size=sample_size,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                top_k=1
            )
            print(f"✅ Saved: {filepath}")
