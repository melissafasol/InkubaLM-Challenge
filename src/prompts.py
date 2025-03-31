import pandas as pd

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