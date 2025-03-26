import random
import warnings
warnings.filterwarnings("ignore")
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, concatenate_datasets, Dataset, Value
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


def load_dataset_by_tag(dataset_type, tag, split='train'):
    return load_dataset(f"{dataset_type}{tag}", split=split)

def load_and_combine_datasets(tag, split='train'):
    """
    Load and combine multiple datasets with common columns.
    
    Args:
        tag (str): Tag for the datasets (Train, Test)
        split (str): Split to load (train, test)
        
    Returns:
        Dataset: Combined dataset with common columns
    """
    # Load datasets with the given split
    se_dataset = load_dataset_by_tag("lelapa/Sentiment", tag, split)
    mt_dataset = load_dataset_by_tag("lelapa/MT", tag, split)
    xn_dataset = load_dataset_by_tag("lelapa/XNLI", tag, split)

    # Identify common columns
    common_columns = list(set(se_dataset.column_names) & 
                          set(mt_dataset.column_names) & 
                          set(xn_dataset.column_names))
    print(f"Common Columns: {common_columns}")

    # Keep only common columns
    se_dataset = se_dataset.remove_columns([col for col in se_dataset.column_names 
                                            if col not in common_columns])
    mt_dataset = mt_dataset.remove_columns([col for col in mt_dataset.column_names 
                                            if col not in common_columns])
    xn_dataset = xn_dataset.remove_columns([col for col in xn_dataset.column_names 
                                            if col not in common_columns])

    # Convert 'targets' column to string type
    se_dataset = se_dataset.cast_column("targets", Value("string"))
    mt_dataset = mt_dataset.cast_column("targets", Value("string"))
    xn_dataset = xn_dataset.cast_column("targets", Value("string"))

    # Concatenate datasets
    combined_dataset = concatenate_datasets([se_dataset, mt_dataset, xn_dataset])

    return combined_dataset


def extract_task_from_id(id_string):
    """
    Extract task type from ID string.
    
    Args:
        id_string (str): ID string containing task information
        
    Returns:
        str: Extracted task type
    """
    task = id_string.split('_')[3]
    # Handle special case for sentiment task
    return 'sentiment' if task == ' dev' else task


def analyze_task_lengths(df, task_column='task'):
    """
    Analyze and display target sequence lengths for each task.
    
    Args:
        df (DataFrame): DataFrame containing task and targets columns
        task_column (str): Name of the task column
        
    Returns:
        dict: Statistics about target lengths per task
    """
    stats = {}
    print("Target sequence length analysis by task:")
    print("="*50)
    
    for task in df[task_column].unique():
        mask = df[task_column] == task
        task_stats = df.loc[mask, 'targets'].apply(lambda x: len(x.split())).describe()
        print(f"Task: {task}")
        print(task_stats)
        print("-"*50)
        stats[task] = task_stats
    
    return stats


def balance_target_lengths(df, task_column='task', reference_task='mt', repetition_factor=11):
    """
    Balance target sequence lengths by repeating shorter targets.
    
    Args:
        df (DataFrame): DataFrame containing task and targets columns
        task_column (str): Name of the task column
        reference_task (str): Task with longer sequences to use as reference
        repetition_factor (int): Number of times to repeat shorter sequences
        
    Returns:
        DataFrame: DataFrame with balanced target lengths
    """
    df_balanced = df.copy()
    
    for task in df_balanced[task_column].unique():
        if task != reference_task:
            mask = df_balanced[task_column] == task
            df_balanced.loc[mask, 'targets'] = df_balanced.loc[mask, 'targets'].apply(
                lambda x: ' '.join([x] * repetition_factor)
            )
    
    return df_balanced


def plot_target_lengths(df_before, df_after, task_column='task'):
    """
    Plot target lengths before and after balancing.
    
    Args:
        df_before (DataFrame): DataFrame before balancing
        df_after (DataFrame): DataFrame after balancing
        task_column (str): Name of the task column
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before balancing
    task_stats_before = {}
    for task in df_before[task_column].unique():
        mask = df_before[task_column] == task
        lengths = df_before.loc[mask, 'targets'].apply(lambda x: len(x.split()))
        task_stats_before[task] = lengths.mean()
    
    # After balancing
    task_stats_after = {}
    for task in df_after[task_column].unique():
        mask = df_after[task_column] == task
        lengths = df_after.loc[mask, 'targets'].apply(lambda x: len(x.split()))
        task_stats_after[task] = lengths.mean()
    
    # Plotting
    ax1.bar(task_stats_before.keys(), task_stats_before.values())
    ax1.set_title('Average Target Length Before Balancing')
    ax1.set_ylabel('Average number of tokens')
    
    ax2.bar(task_stats_after.keys(), task_stats_after.values())
    ax2.set_title('Average Target Length After Balancing')
    
    plt.tight_layout()
    #plt.savefig('target_length_comparison.png')
    plt.show()
    #plt.close()


def formatting_prompts_func(example):
    """
    Format examples for instruction tuning.
    
    Args:
        example (dict): Example containing instruction, inputs, and targets
        
    Returns:
        str: Formatted prompt
    """
    if example['targets'] is not None:
        return f"### Instruction: {example['instruction']}\n### Input: {example['inputs']}\n### Response: {example['targets']}"
    return f"### Instruction: {example['instruction']}\n### Input: {example['inputs']}\n### Response:"


import os

def setup_model_and_tokenizer(model_name, use_4bit=True):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    hf_token = os.environ.get("HF_TOKEN", None)  # 👈 fetch token from env or default

    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token if hf_token and hf_token != "----" else None,
    )

    return model, tokenizer, bnb_config



def apply_lora_adapters(model, r=8, lora_alpha=16, dropout=0.05):
    """
    Apply LoRA adapters to the model.
    
    Args:
        model: Base model
        r (int): LoRA rank
        lora_alpha (int): LoRA alpha parameter
        dropout (float): Dropout probability
        
    Returns:
        model: Model with LoRA adapters
    """
    # Define LoRA Config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA adapters to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def setup_trainer(model, dataset, tokenizer, output_dir, num_epochs=3):
    """
    Set up SFTTrainer for fine-tuning.
    
    Args:
        model: Model with LoRA adapters
        dataset: Training dataset
        tokenizer: Tokenizer
        output_dir (str): Output directory for checkpoints
        num_epochs (int): Number of training epochs
        
    Returns:
        SFTTrainer: Trainer object
    """
    # Define response template for proper label masking
    response_template_with_context = "\n### Response:"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    
    # Data collator for masked LM training
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    
    # Training arguments
    train_args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=512,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        report_to=[],  # Disable wandb
    )
    
    # Trainer setup
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=train_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    
    return trainer


def generate_response(model, tokenizer, prompt, max_new_tokens=20):
    """
    Generate response using fine-tuned model.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        prompt (str): Input prompt
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: Generated response
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    return response


def encode_sentiment_label(label):
    """
    Encode sentiment label to integer.
    
    Args:
        label (str): Sentiment label
        
    Returns:
        int: Encoded label
    """
    for c, i in enumerate(["Chanya", "Wastani", "Hasi"]):
        if label == i:
            return c
    for c, i in enumerate(["Kyakkyawa", "Tsaka-tsaki", "Korau"]):
        if label == i:
            return c
    return 0


def apply_inference_to_test_data(model, tokenizer, test_dataset):
    """
    Apply inference to test dataset.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        test_dataset: Test dataset
        
    Returns:
        DataFrame: DataFrame with generated responses
    """
    df = pd.DataFrame(test_dataset)
    model.eval()
    
    # Apply inference with tqdm progress bar
    tqdm.pandas(desc="Generating Responses")
    df['generated'] = df.progress_apply(
        lambda row: generate_response(model, tokenizer, formatting_prompts_func(row)), 
        axis=1
    )
    
    # Process responses based on task type
    df['Response'] = ''
    
    # Sentiment task
    mask = df.ID.apply(lambda x: 'sentiment' in x)
    df.loc[mask, 'Response'] = df.loc[mask, 'generated'].apply(
        lambda x: encode_sentiment_label(x.strip().split()[0])
    )
    
    # XNLI task
    mask = df.ID.apply(lambda x: 'afrixnli' in x)
    df.loc[mask, 'Response'] = df.loc[mask, 'generated'].apply(
        lambda x: int(x.strip().split()[0])%3 if x.strip().split()[0].isdigit() else 0
    )
    
    # MT task
    mask = df.ID.apply(lambda x: 'mt_' in x)
    df.loc[mask, 'Response'] = df.loc[mask, 'generated']
    
    return df

def display_formatted_examples(df, num_examples=2):
    """
    Display formatted examples for each task.
    
    Args:
        df (DataFrame): DataFrame containing the examples
        num_examples (int): Number of examples to display per task
    """
    for task in df.task.unique():
        print(f"\n\n{'='*40}\nTask: {task}\n{'='*40}")
        mask = df.task == task
        for i, (_, row) in enumerate(df[mask].iterrows()):
            if i >= num_examples:
                break
                
            print(f"\nExample {i+1}:")
            print("-" * 40)
            formatted = formatting_prompts_func(row)
            print(formatted)
            print("-" * 40)
            

def balance_target_lengths(
    df,
    task_column='task',
    target_column='targets',
    reference_task='mt',
    repetition_factor=None,
    tokenizer=str.split  # Tokenizer function, defaults to whitespace split
):
    """
    Balance target sequence lengths by repeating shorter targets.

    Args:
        df (DataFrame): DataFrame containing task and targets columns
        task_column (str): Name of the task column
        target_column (str): Name of the target column
        reference_task (str): Task with longer sequences to use as reference
        repetition_factor (int or None): If set, uses fixed repetition. If None, uses dynamic repetition based on average length of reference task.
        tokenizer (callable): Tokenizer function to estimate token lengths. Defaults to str.split (whitespace split).

    Returns:
        DataFrame: DataFrame with balanced target lengths
    """
    df_balanced = df.copy()

    # Compute average length of reference task targets
    ref_mask = df_balanced[task_column] == reference_task
    reference_lengths = df_balanced.loc[ref_mask, target_column].apply(lambda x: len(tokenizer(x)))
    avg_ref_len = reference_lengths.mean()

    for task in df_balanced[task_column].unique():
        if task != reference_task:
            mask = df_balanced[task_column] == task
            if repetition_factor is not None:
                # Fixed repetition
                df_balanced.loc[mask, target_column] = df_balanced.loc[mask, target_column].apply(
                    lambda x: ' '.join([x] * repetition_factor)
                )
            else:
                # Dynamic repetition to match reference avg length
                df_balanced.loc[mask, target_column] = df_balanced.loc[mask, target_column].apply(
                    lambda x: (x + ' ') * max(1, round(avg_ref_len / len(tokenizer(x))))
                )

    return df_balanced

def setup_trainer_ab_testing(
    model, 
    dataset, 
    tokenizer, 
    output_dir, 
    num_epochs=3, 
    val_dataset=None,
    log_token_lengths=False,
    log_per_task_loss=False
):
    """
    Set up SFTTrainer for fine-tuning with optional logging and evaluation.

    Args:
        model: Model with LoRA adapters
        dataset: Training dataset
        tokenizer: Tokenizer
        output_dir (str): Output directory for checkpoints
        num_epochs (int): Number of training epochs
        val_dataset (Dataset, optional): Validation dataset for evaluation
        log_token_lengths (bool): If True, print avg token length per task
        log_per_task_loss (bool): If True, logs loss by task to console (lightweight)

    Returns:
        SFTTrainer: Configured trainer
    """

    # Optional token length logging
    if log_token_lengths:
        dataset_df = dataset.to_pandas()
        dataset_df["token_len"] = dataset_df["targets"].apply(lambda x: len(tokenizer.tokenize(x)))
        print("\n🔍 Token length stats by task:")
        print(dataset_df.groupby("task")["token_len"].describe())

    # Optional task grouping for per-task logging (lightweight)
    if log_per_task_loss:
        print("\n🧪 Note: to log true per-task loss you'd need to modify the loss function with labels.")
        print("This option is for prep only — actual per-task logging requires a custom training loop.")

    # Define response template for proper label masking
    response_template_with_context = "\n### Response:"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]

    # Data collator for causal LM masking
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    train_args = SFTConfig(
    output_dir=output_dir,
    max_seq_length=512,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir=f"{output_dir}/logs",  # 👈 ensure logs directory is defined
    logging_steps=10,                  # 👈 print every 10 steps
    save_total_limit=2,
    report_to="none",                  # 👈 prevent WandB if not used
    disable_tqdm=False,                # 👈 make sure progress bar prints
    logging_first_step=True,          # 👈 log first step no matter what
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch" if val_dataset else "no",
    load_best_model_at_end=True if val_dataset else False)

   

    # Trainer setup
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        args=train_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    return trainer
