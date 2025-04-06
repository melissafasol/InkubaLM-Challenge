import os 
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
from torch.nn import CrossEntropyLoss


def load_dataset_by_tag(dataset_type, tag, split='train'):
    return load_dataset(f"{dataset_type}{tag}", split=split)

def load_and_combine_datasets(tag, split='train'):
    """
    Load and combine multiple datasets with all unique columns (union).
    Empty strings are used for missing values.
    
    Args:
        tag (str): Tag for the datasets (Train, Test)
        split (str): Split to load (train, test)
        
    Returns:
        Dataset: Combined dataset with all unique columns
    """
    # Load datasets with the given split
    se_dataset = load_dataset_by_tag("lelapa/Sentiment", tag, split)
    mt_dataset = load_dataset_by_tag("lelapa/MT", tag, split)
    xn_dataset = load_dataset_by_tag("lelapa/XNLI", tag, split)

    # Identify all unique columns (union)
    all_columns = list(set(se_dataset.column_names) | 
                      set(mt_dataset.column_names) | 
                      set(xn_dataset.column_names))
    print(f"All Columns: {all_columns}")

    # Function to ensure dataset has all columns, filling missing ones with empty strings
    def ensure_all_columns(dataset, all_cols):
        # Add each missing column one by one
        for col in all_cols:
            if col not in dataset.column_names:
                # Create array of empty strings with the same length as the dataset
                empty_column = [""] * len(dataset)
                dataset = dataset.add_column(col, empty_column)
        
        return dataset

    # Ensure all datasets have all columns
    se_dataset = ensure_all_columns(se_dataset, all_columns)
    mt_dataset = ensure_all_columns(mt_dataset, all_columns)
    xn_dataset = ensure_all_columns(xn_dataset, all_columns)

    # Make sure 'targets' column is string type if it exists in all datasets
    if "targets" in all_columns:
        se_dataset = se_dataset.cast_column("targets", Value("string"))
        mt_dataset = mt_dataset.cast_column("targets", Value("string"))
        xn_dataset = xn_dataset.cast_column("targets", Value("string"))

    return se_dataset, mt_dataset, xn_dataset


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
    premise = example['premise']
    premise = premise+'\n' if len(premise) else ''
    if example['targets'] is not None:
        return f"### Instruction: {example['instruction']}\n### Input: {premise}{example['inputs']}\n### Response: {example['targets']}"
    return f"### Instruction: {example['instruction']}\n### Input: {premise}{example['inputs']}\n### Response:"


def setup_model_and_tokenizer(model_name, token, use_4bit=True):
    """
    Set up model and tokenizer for QLoRA fine-tuning.
    
    Args:
        model_name (str): Name of the base model
        use_4bit (bool): Whether to use 4-bit quantization
        
    Returns:
        tuple: (model, tokenizer, bnb_config)
    """
    # Define BitsAndBytes config for quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=token,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    
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
    Set up SFTTrainer for direct fine-tuning.
    
    Args:
        model: Model to fine-tune
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
        max_seq_length=256,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        optim = 'adamw_bnb_8bit',
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
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
