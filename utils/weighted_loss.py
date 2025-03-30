import torch
import os
from collections import Counter
from trl import SFTTrainer, SFTConfig
from torch.nn import CrossEntropyLoss
from trl.trainer import DataCollatorForCompletionOnlyLM


class WeightedSFTTrainer(SFTTrainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # Create loss function with class weights (if given)
        loss_fct = CrossEntropyLoss(weight=self.class_weights) if self.class_weights is not None else CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss



def get_class_weights(dataset, class_names=("Chanya", "Wastani", "Hasi")):
    label_counts = Counter(dataset["targets"])
    weights = [1.0 / label_counts[label] for label in class_names]
    weights = torch.tensor(weights)
    return weights / weights.sum()  # normalize

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



def setup_trainer(model, dataset, tokenizer, output_dir, num_epochs=3, lang="swahili"):
    """
    Set up WeightedSFTTrainer for fine-tuning with class weighting.
    """
    
    # Tokenize the response marker
    response_template_with_context = "\n### Response:"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]

    # Set up collator
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # Define class names per language
    if lang == "swahili":
        class_names = ["Chanya", "Wastani", "Hasi"]
    elif lang == "hausa":
        class_names = ["Kyakkyawa", "Tsaka-tsaki", "Korau"]
    else:
        raise ValueError(f"Unsupported language: {lang}")

    # Compute weights
    weights = get_class_weights(dataset, class_names=class_names)
    weights = weights.to(model.device)

    # Training config
    train_args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=512,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        report_to=[],
    )

    # Use Weighted Trainer
    trainer = WeightedSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=train_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        class_weights=weights,
    )

    return trainer
