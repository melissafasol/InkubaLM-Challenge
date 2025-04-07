import matplotlib.pyplot as plt

def plot_target_lengths(df_before, df_after, task_column='task'):
    """
    Pretty plot of average target lengths before and after token balancing.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Define tasks and colors
    tasks = sorted(df_before[task_column].unique())
    colors = plt.cm.Set2.colors[:len(tasks)]
    color_map = {task: colors[i] for i, task in enumerate(tasks)}

    # Compute average token lengths
    lengths_before = {
        task: df_before[df_before[task_column] == task]['targets']
        .apply(lambda x: len(str(x).split()))
        .mean()
        for task in tasks
    }

    lengths_after = {
        task: df_after[df_after[task_column] == task]['targets']
        .apply(lambda x: len(str(x).split()))
        .mean()
        for task in tasks
    }

    # Plot BEFORE
    ax1.bar(lengths_before.keys(), lengths_before.values(),
            color=[color_map[task] for task in tasks])
    ax1.set_title("Average target length\nbefore balancing", fontsize=16)
    ax1.set_ylabel("Avg number of tokens", fontsize=12)
    ax1.set_xlabel("Task", fontsize=12)

    # Plot AFTER
    ax2.bar(lengths_after.keys(), lengths_after.values(),
            color=[color_map[task] for task in tasks])
    ax2.set_title("Average target length\nafter balancing", fontsize=16)
    ax2.set_xlabel("Task", fontsize=12)

    # Clean look: remove bounding boxes
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', labelrotation=10, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
