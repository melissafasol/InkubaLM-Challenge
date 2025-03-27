import csv
import time
import torch
from transformers import AutoModelForCausalLM

# Function to load the model for text generation
def load_model(model_name):
    """
    Load a causal language model with automatic GPU support.
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
    """
    Runs inference on a dataset and logs responses and log-likelihoods to a CSV file.

    Parameters:
    - model: The model or trainer object.
    - tokenizer: Tokenizer for the model.
    - BASE_PROMPT: Prompt format string.
    - task_instruction: Optional custom instruction.
    - dataset: The dataset to run inference on.
    - csv_file_path: Path to output CSV file.
    - custom_instruct: Whether to override dataset instructions.
    - sample_size: Number of samples to evaluate.
    - max_new_tokens, do_sample, temperature, etc.: Generation settings.
    """
    actual_model = getattr(model, "model", model)
    actual_model.eval()

    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "ID",
            "Instruction",
            "Input Text",
            "Response",
            "Log-Likelihood",
            "Targets",
            "Task",
            "Langs",
        ])

        for i, item in enumerate(dataset):
            if i >= sample_size:
                break

            instruction = item["instruction"] if not custom_instruct else task_instruction
            input_text = item["inputs"]
            labels = item["targets"]
            langs = item["langs"]
            task = item.get("task", "xnli")
            identity = item["ID"]

            user_prompt = BASE_PROMPT.format(f"{instruction}\n{input_text}")
            batch = tokenizer(user_prompt, return_tensors="pt")
            batch = {k: v.to(actual_model.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = actual_model.generate(
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

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(user_prompt):]

            # Compute log-likelihoods for classification tasks
            if task != "mmt":
                with torch.no_grad():
                    logits = actual_model(**batch).logits
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                    t_labels = (
                        torch.tensor([0, 1, 2])
                        .unsqueeze(0).unsqueeze(0)
                        .expand(batch["input_ids"].size(0), batch["input_ids"].size(1), -1)
                        .to(actual_model.device)
                    )

                    log_likelihoods_per_class = log_probs.gather(2, t_labels).mean(dim=1)
            else:
                log_likelihoods_per_class = []

            writer.writerow([
                identity,
                instruction,
                input_text,
                output_text,
                log_likelihoods_per_class.tolist() if isinstance(log_likelihoods_per_class, torch.Tensor) else [],
                labels,
                task,
                langs,
            ])
