import re
import time
import csv 
import torch 


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
    top_k=50,
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
