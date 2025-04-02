'''Model setup, LoRA adapters, training setup'''

import os 
import torch
import random
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, concatenate_datasets, Dataset, Value
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from torch.nn import CrossEntropyLoss



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
    save_strategy="steps",                # 👈 match eval strategy!
    save_steps=10,                        # 👈 how often to save
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    eval_steps=10,
    save_total_limit=2,
    report_to="none",
    disable_tqdm=False,
    logging_first_step=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    load_best_model_at_end=True
)


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

import csv
import time

import torch
from transformers import AutoModelForCausalLM


# Function to load the model for text generation
def load_model(model_name):
    """
    A function to load the model, it allows for GPU to be used if it is available
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    return model


# Helper to clean up model outputs
def clean_output(text):
    # Cut off anything before/after the actual translation
    if "Hausa:" in text:
        text = text.split("Hausa:")[-1]

    # Remove prompt leakage
    text = re.sub(r"below is an instruction.*", "", text, flags=re.IGNORECASE)

    # Remove emojis and corrupted unicode
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Remove excessive repetition like "ewa, ewa"
    text = re.sub(r"\b(ẽwa[,\.]?\s*){2,}", " ", text, flags=re.IGNORECASE)

    return text.strip()

# Run inference for a specific task
def main(
    model,
    tokenizer,
    BASE_PROMPT,
    task_instruction,
    dataset,
    csv_file_path,
    custom_instruct=False,
    sample_size=4,
    max_new_tokens=100,
    seed=42,
    do_sample=True,
    min_length=None,
    use_cache=True,
    top_p=1.0,
    temperature=0.5,
    top_k=5,
    repetition_penalty=1.2,
    length_penalty=1,
    **kwargs,
):
    actual_model = getattr(model, "model", model)
    actual_model.eval()

    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "ID",
                "Instruction",
                "Input Text",
                "Response",
                "Log-Likelihood",
                "Targets",
                "Task",
                "Langs",
            ]
        )

        for i, item in enumerate(dataset):
            if i >= sample_size:
                break

            if not custom_instruct:
                instruction = item["instruction"]
            else:
                instruction = task_instruction
            input_text = item["inputs"]
            labels = item["targets"]
            langs = item["langs"]
            task = item.get("task", "xnli")
            identity = item["ID"]

            user_prompt = BASE_PROMPT.format(f"{instruction}\n{input_text}")
            batch = tokenizer(user_prompt, return_tensors="pt")
            batch = {k: v.to(model.device) for k, v in batch.items()}

            start = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    min_length=min_length,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    **kwargs,
                )

            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = clean_output(raw_output[len(user_prompt):])

            # Optional: log weird outputs
            if re.search(r"[�💖]", output_text):
                print(f"⚠️ Strange output for ID {identity}: {output_text[:100]}")

            if task != "mmt":
                with torch.no_grad():
                    logits = model(**batch).logits
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                    t_labels = (
                        torch.tensor([0, 1, 2])
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand(batch["input_ids"].size(0), batch["input_ids"].size(1), -1)
                        .to(model.device)
                    )

                    log_likelihoods_per_class = log_probs.gather(2, t_labels).mean(dim=1)
            else:
                log_likelihoods_per_class = []

            writer.writerow(
                [
                    identity,
                    instruction,
                    input_text,
                    output_text,
                    log_likelihoods_per_class,
                    labels,
                    task,
                    langs,
                ]
            )
