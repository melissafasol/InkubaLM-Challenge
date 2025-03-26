from trl import SFTTrainer, SFTConfig
from trl.trainer import ConstantLengthDataset
from transformers import DataCollatorForSeq2Seq
from datasets import DatasetDict
from collections import defaultdict
import numpy as np
from multitask import formatting_prompts_func

def setup_trainer(
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

    # Training arguments
    train_args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=512,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        report_to=[],  # Disable wandb
        evaluation_strategy="epoch" if val_dataset else "no",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        load_best_model_at_end=True if val_dataset else False,
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
