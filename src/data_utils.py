'''Data loading, balancing and preprocessing '''
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset, Value
import matplotlib.pyplot as plt

def load_dataset_by_tag(dataset_type, tag, split='train'):
    return load_dataset(f"{dataset_type}{tag}", split=split)

def load_and_combine_datasets(tag, split='train'):
    """
    Load and combine multiple datasets with common columns.
    
    Args:
        tag (str): Tag for the datasets (Train, Test)
        split (str): Split to load (train, test)
        
    Returns:
        Dataset: Combined dataset with common columns
    """
    # Load datasets with the given split
    se_dataset = load_dataset_by_tag("lelapa/Sentiment", tag, split)
    mt_dataset = load_dataset_by_tag("lelapa/MT", tag, split)
    xn_dataset = load_dataset_by_tag("lelapa/XNLI", tag, split)

    # Identify common columns
    common_columns = list(set(se_dataset.column_names) & 
                          set(mt_dataset.column_names) & 
                          set(xn_dataset.column_names))
    print(f"Common Columns: {common_columns}")

    # Keep only common columns
    se_dataset = se_dataset.remove_columns([col for col in se_dataset.column_names 
                                            if col not in common_columns])
    mt_dataset = mt_dataset.remove_columns([col for col in mt_dataset.column_names 
                                            if col not in common_columns])
    xn_dataset = xn_dataset.remove_columns([col for col in xn_dataset.column_names 
                                            if col not in common_columns])

    # Convert 'targets' column to string type
    se_dataset = se_dataset.cast_column("targets", Value("string"))
    mt_dataset = mt_dataset.cast_column("targets", Value("string"))
    xn_dataset = xn_dataset.cast_column("targets", Value("string"))

    # Concatenate datasets
    combined_dataset = concatenate_datasets([se_dataset, mt_dataset, xn_dataset])

    return combined_dataset

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


def balance_target_lengths(
    df,
    task_column='task',
    target_column='targets',
    reference_task='mt',
    repetition_factor=None,
    tokenizer=str.split
):
    """
    Dynamically or statically balance target lengths per task.
    """
    df_balanced = df.copy()

    # Compute average token length of reference task
    ref_mask = df_balanced[task_column] == reference_task
    avg_ref_len = df_balanced.loc[ref_mask, target_column].apply(lambda x: len(tokenizer(x))).mean()

    for task in df_balanced[task_column].unique():
        if task != reference_task:
            mask = df_balanced[task_column] == task
            if repetition_factor is not None:
                df_balanced.loc[mask, target_column] = df_balanced.loc[mask, target_column].apply(
                    lambda x: ' '.join([x] * repetition_factor)
                )
            else:
                df_balanced.loc[mask, target_column] = df_balanced.loc[mask, target_column].apply(
                    lambda x: (x + ' ') * max(1, round(avg_ref_len / len(tokenizer(x))))
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
    
