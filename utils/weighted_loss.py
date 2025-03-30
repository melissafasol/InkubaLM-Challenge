import torch
import os
from collections import Counter
from trl import SFTTrainer, SFTConfig
from torch.nn import CrossEntropyLoss
from trl.trainer import DataCollatorForCompletionOnlyLM


class MultiTaskWeightedSFTTrainer(SFTTrainer):
    def __init__(self, task_class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Normalize weights to sum to 1
        self.task_class_weights = {
            task: weights / weights.sum() for task, weights in (task_class_weights or {}).items()
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        task_names = inputs.get("task", None)

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        task = task_names[0] if isinstance(task_names, (list, torch.Tensor)) else task_names
        task = task if isinstance(task, str) else task.item() if task is not None else None

        weights = self.task_class_weights.get(task, None)
        loss_fct = CrossEntropyLoss(weight=weights.to(model.device) if weights is not None else None)

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss



def get_class_weights(dataset, class_names):
    label_counts = Counter(dataset["targets"])
    weights = [1.0 / label_counts.get(label, 1) for label in class_names]
    return torch.tensor(weights)


def formatting_prompts_func(example):
    if example['targets'] is not None:
        return f"### Instruction: {example['instruction']}\n### Input: {example['inputs']}\n### Response: {example['targets']}"
    return f"### Instruction: {example['instruction']}\n### Input: {example['inputs']}\n### Response:"


def setup_trainer(model, dataset, tokenizer, output_dir, num_epochs=3):
    """
    Set up MultiTaskWeightedSFTTrainer for multitask fine-tuning.
    """
    response_template_with_context = "\n### Response:"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # Define class label names per task (adjust as needed)
    task_label_map = {
        "sentiment_swahili": ["Chanya", "Wastani", "Hasi"],
        "sentiment_hausa": ["Kyakkyawa", "Tsaka-tsaki", "Korau"],
        "xnli": ["0", "1", "2"],  # use strings if targets are strings
    }

    # Compute task-specific class weights
    task_class_weights = {}
    for task_key, labels in task_label_map.items():
        # Filter dataset by task
        task_data = dataset.filter(lambda x: x["task"] == task_key)
        if len(task_data) > 0:
            task_class_weights[task_key] = get_class_weights(task_data, class_names=labels)

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

    trainer = MultiTaskWeightedSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=train_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        task_class_weights=task_class_weights,
    )

    return trainer
