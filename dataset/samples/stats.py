import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_diff_length_combined_distribution_plot(dfs, labels, colors, output_path):
    """
    This graph can be combined distribution plots for multiple DataFrames:
    - One subplot for diff word count
    - One subplot for diff character count
    Each DataFrame is plotted in a different color.
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    for df, label, color in zip(dfs, labels, colors):
        # Calculate word and character counts from 'diff' column
        df['diff_words_count'] = df['diff'].apply(lambda x: len(str(x).split()))
        df['diff_chars_count'] = df['diff'].apply(lambda x: len(str(x)))

        # Plot word count
        sns.kdeplot(df['diff_words_count'], ax=axes[0], label=label, color=color, fill=True, alpha=0.3)
        # Plot character count
        sns.kdeplot(df['diff_chars_count'], ax=axes[1], label=label, color=color, fill=True, alpha=0.3)

    # Axis titles and labels
    axes[0].set_title('Diff Word Count Distribution (samples)')
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    axes[1].set_title('Diff Character Count Distribution (samples)')
    axes[1].set_xlabel('Character Count')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()




def generate_commit_message_combined_distribution_plot(dfs, labels, colors, output_path):
    """
    Generate combined distribution plots for multiple DataFrames:
    - One subplot for commit message word count
    - One subplot for commit message character count
    Each DataFrame is plotted in a different color.
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    for df, label, color in zip(dfs, labels, colors):
        # Compute word and character counts for commit messages
        df['message_words_count'] = df['message'].apply(lambda x: len(str(x).split()))
        df['message_chars_count'] = df['message'].apply(lambda x: len(str(x)))


        # Plot word count
        sns.kdeplot(df['message_words_count'], ax=axes[0], label=label, color=color, fill=True, alpha=0.3)
        # Plot character count
        sns.kdeplot(df['message_chars_count'], ax=axes[1], label=label, color=color, fill=True, alpha=0.3)


    # Customize plot 1: Word count
    axes[0].set_title('Commit Message Word Count Distribution (samples)')
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Customize plot 2: Character count
    axes[1].set_title('Commit Message Character Count Distribution (samples)')
    axes[1].set_xlabel('Character Count')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()

if __name__ == "__main__":
    # Check if the script is being run directly
    print("Generating stats for the samples")

    csv_files = [
       "dataset/samples/java.csv",
       "dataset/samples/js.csv",
       "dataset/samples/php.csv",
       "dataset/samples/py.csv",
    ]

    # Load each CSV into a DataFrame
    dfs = [pd.read_csv(file) for file in csv_files]

    # Extract just the filename (without path or .csv) for labeling
    labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

    # Colors for each language
    colors = ['blue', 'orange', 'green', 'purple']


    # Generate and save the combined diff length plot
    generate_diff_length_combined_distribution_plot(
        dfs=dfs,
        labels=labels,
        colors=colors,
        output_path="dataset/samples/diff_length_distribution.png"
    )

    
    # Generate and save the combined commit message length plot
    generate_commit_message_combined_distribution_plot(
        dfs=dfs,
        labels=labels,
        colors=colors,
        output_path="dataset/samples/commit_message_length_distribution.png"
    )