import torch 
import pandas as pd
from tqdm.auto import tqdm

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
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.2
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