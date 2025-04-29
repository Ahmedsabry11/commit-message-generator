# read csv file and print the number of rows and columns
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv('dataset/commitbench.csv')

# Print the number of rows and columns
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# print header of the dataset
print(df.head())

print(df.columns)

# Drop rows with missing language info if needed
df = df.dropna(subset=['diff_languages'])

# Target languages (normalized)
target_languages = {
    'py': 'dataset/py.csv',
    'java': 'dataset/java.csv',
    'php': 'dataset/php.csv',
    'js': 'dataset/js.csv'
}

# Iterate over each target language
for lang, filename in target_languages.items():
    # Filter rows where the language appears in the list
    lang_df = df[df['diff_languages'].str.contains(rf'\b{lang}\b', regex=True)]

    # Save to a CSV file
    if not lang_df.empty:
        lang_df.to_csv(filename, index=False)
        print(f"Saved {filename}")