import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_diff_length_distribution_plot(df,output_path):
    """
    Generate two separate subplots showing the distribution of diff lengths:
    one for word count and one for character count.
    """
    # Compute word and character counts from diff
    df['diff_words_count'] = df['diff'].apply(lambda x: len(x.split()))
    df['diff_chars_count'] = df['diff'].apply(len)

    # Create figure and subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # Plot 1: Word count distribution
    sns.histplot(df['diff_words_count'], bins=30, kde=True, color='blue', ax=axes[0])
    axes[0].set_title('Diff Word Count Distribution')
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Frequency')

    # Plot 2: Character count distribution
    sns.histplot(df['diff_chars_count'], bins=30, kde=True, color='orange', ax=axes[1])
    axes[1].set_title('Diff Character Count Distribution')
    axes[1].set_xlabel('Character Count')
    axes[1].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()

def generate_commit_message_length_distribution_plot(df,output_path):
    """
    Generate two separate subplots showing the distribution of commit message lengths:
    one for word count and one for character count.
    """
    # Compute word and character counts
    df['message_words_count'] = df['message'].apply(lambda x: len(x.split()))
    df['message_chars_count'] = df['message'].apply(len)

    # Create figure and subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # Plot 1: Word count distribution
    sns.histplot(df['message_words_count'], bins=30, kde=True, color='blue', ax=axes[0])
    axes[0].set_title('Commit Message Word Count Distribution')
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Frequency')

    # Custom x-ticks for word count (every 10 words)
    max_words = df['message_words_count'].max()
    axes[0].set_xticks(np.arange(0, max_words + 10, 10))


    # Plot 2: Character count distribution
    sns.histplot(df['message_chars_count'], bins=30, kde=True, color='orange', ax=axes[1])
    axes[1].set_title('Commit Message Character Count Distribution')
    axes[1].set_xlabel('Character Count')
    axes[1].set_ylabel('Frequency')

    # Custom x-ticks for character count (every 100 chars)
    max_chars = df['message_chars_count'].max()
    axes[1].set_xticks(np.arange(0, max_chars + 100, 100))


    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()

def plot_diff_language_distribution(df, output_path, top_n=20):
    """
    Plot the distribution of the top N programming languages in the diff_languages column.
    """
    # Get top N languages
    top_langs = df['diff_languages'].value_counts().nlargest(top_n).index

    # Replace less frequent with 'Other'
    df['diff_languages_limited'] = df['diff_languages'].apply(lambda x: x if x in top_langs else 'Other')

    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=df,
        x='diff_languages_limited',
        order=df['diff_languages_limited'].value_counts().index,
        palette='Set2'
    )

    plt.title(f'Distribution of Top {top_n} Diff Languages')
    plt.xlabel('Language')
    plt.ylabel('Number of Diffs')

    # Rotate for readability
    plt.xticks(rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()

    
def plot_diff_language_distribution(df, output_path, top_n=20):
    """
    Plot the distribution of the top N programming languages in the diff_languages column.
    """
    # Get top N languages
    top_langs = df['diff_languages'].value_counts().nlargest(top_n).index

    # Replace less frequent with 'Other'
    df['diff_languages_limited'] = df['diff_languages'].apply(lambda x: x if x in top_langs else 'Other')

    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=df,
        x='diff_languages_limited',
        order=df['diff_languages_limited'].value_counts().index,
        palette='Set2'
    )

    plt.title(f'Distribution of Top {top_n} Diff Languages')
    plt.xlabel('Language')
    plt.ylabel('Number of Diffs')

    # Rotate for readability
    plt.xticks(rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()

if __name__ == "__main__":
    # Check if the script is being run directly
    print("Running stats.py directly")

    # Load the dataset
    fileName = "dataset/commitbench.csv"
    df = pd.read_csv(fileName)

    # Print the number of rows and columns
    print("Dataset Loaded Successfully!")
    print("==" * 20)
    print("Dataset Statistics:")
    print("==" * 20)

    # Shape of the dataset
    print(f"Dataset shape: {df.shape}")

    # Print Columns Names
    print("Columns in the dataset:")
    print(df.columns) # Index(['hash', 'diff', 'message', 'project', 'split', 'diff_languages'], dtype='object')    
    
    # Generate and save the combined plot
    generate_diff_length_distribution_plot(df, output_path="dataset/plots/diff_length_distribution.png")

    # Generate commit message length distribution plot
    generate_commit_message_length_distribution_plot(df, output_path="dataset/plots/commit_message_length_distribution.png")

    # Generate distribution of diff languages
    plot_diff_language_distribution(df, output_path="dataset/plots/diff_language_distribution.png")