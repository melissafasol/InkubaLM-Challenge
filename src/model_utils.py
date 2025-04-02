import csv
import time
import torch
from torch.nn.functional import log_softmax


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
    debug_labels=False,  # 👈 Enable to print tokenization info
    **kwargs,
):
    actual_model = getattr(model, "model", model)
    actual_model.eval()

    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "ID",
            "Instruction",
            "Input Text",
            "Response",
            "Log-Likelihoods",
            "Targets",
            "Task",
            "Langs",
        ])

        for i, item in enumerate(dataset):
            if i >= sample_size:
                break

            instruction = item["instruction"] if not custom_instruct else task_instruction
            input_text = item["inputs"]
            target_label = str(item["targets"])
            langs = item["langs"]
            task = item.get("task", "xnli")
            identity = item["ID"]

            if task == "mmt":
                user_prompt = BASE_PROMPT.format(f"{instruction}\n{input_text}")
                batch = tokenizer(user_prompt, return_tensors="pt").to(model.device)

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

                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)[
                    len(user_prompt):
                ].strip()

                response = decoded_output
                log_likelihoods = []

            else:
                # Classification with log-likelihood scoring

                if task == "xnli":
                    if langs == "swa":
                        label_texts = ["Kweli", "Wala siyo", "Uongo"]
                    elif langs == "hau":
                        label_texts = ["Gaskiya", "Tsaka-tsaki", "Karya"]
                    else:
                        label_texts = ["True", "Neither", "False"]
                else:
                    label_texts = ["Kyakkyawa", "Tsaka-tsaki", "Korau"] if "hausa" in langs else ["Chanya", "Wastani", "Hasi"]

                label_to_index = {label: idx for idx, label in enumerate(label_texts)}
                log_likelihoods = []

                for label_text in label_texts:
                    # DEBUG: Print tokenization info
                    if debug_labels:
                        tokenized = tokenizer.tokenize(label_text)
                        print(f"Label: {label_text} → tokens: {tokenized} ({len(tokenized)} tokens)")

                    prompt_text = BASE_PROMPT.format(f"{instruction}\n{input_text}")
                    prompt_tokens = tokenizer(prompt_text, return_tensors="pt").to(model.device)
                    label_tokens = tokenizer(label_text, return_tensors="pt").to(model.device)

                    input_ids = torch.cat([prompt_tokens["input_ids"], label_tokens["input_ids"][:, 1:]], dim=1)
                    attention_mask = torch.cat([prompt_tokens["attention_mask"], label_tokens["attention_mask"][:, 1:]], dim=1)

                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        log_probs = log_softmax(logits, dim=-1)

                    label_token_ids = label_tokens["input_ids"][0][1:]  # remove BOS
                    label_start = input_ids.size(1) - label_token_ids.size(0)
                    token_log_probs = log_probs[0, label_start:label_start + len(label_token_ids)]
                    selected_token_log_probs = token_log_probs.gather(1, label_token_ids.unsqueeze(1)).squeeze(1)

                    # 🔁 Use sum instead of mean
                    normalized_log_prob = selected_token_log_probs.sum().item() / len(label_token_ids)
                    log_likelihoods.append(normalized_log_prob)

                best_index = int(torch.argmax(torch.tensor(log_likelihoods)))
                best_label = label_texts[best_index]
                response = label_to_index[best_label]  # 🔁 Map back to 0/1/2 for Zindi

            writer.writerow([
                identity,
                instruction,
                input_text,
                response,
                log_likelihoods,
                target_label,
                task,
                langs,
            ])
