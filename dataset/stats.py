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


# print data of certain columnn in a sample of one row
print(df.iloc[0]['diff'])

# plot the distribution of the length of the commit messages
df['message_length'] = df['diff'].apply(lambda x: len(x.split()))
print(df['message_length'].describe())
plt.figure(figsize=(10, 6))
sns.histplot(df['message_length'], bins=30, kde=True)
plt.title('Distribution of Commit Message Lengths')
plt.xlabel('Length of Commit Message (in words)')
plt.ylabel('Frequency')
# save the plot
plt.savefig('dataset/commit_message_length_distribution.png')
plt.show()


df['character_length'] = df['diff'].apply(lambda x: len(x))
# Create bins of size 100
df['length_bin'] = (df['character_length'] // 100) * 100

# Group by bin and count
bin_counts = df.groupby('length_bin').size().reset_index(name='count')
# Print the result
print(df['character_length'].describe())
print(bin_counts)

plt.figure(figsize=(10, 6))
sns.barplot(data=bin_counts, x='length_bin', y='count', color='skyblue')
plt.title('Frequency of Commit Diffs Grouped by Length (Every 100 Characters)')
plt.xlabel('Commit Diff Length (Grouped, Lower Bound of Bin)')
plt.ylabel('Number of Commits')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('dataset/commit_length_bins.png')
plt.show()

# plt.figure(figsize=(10, 6))
# sns.histplot(df['character_length'], bins=30, kde=True)
# plt.title('Distribution of Commit Message Character Lengths')
# plt.xlabel('Length of Commit Message (in characters)')
# plt.ylabel('Frequency')
# # plt.xlim(0, 1000)  # adjust depending on your use case
# # save the plot
# plt.savefig('dataset/commit_message_character_length_distribution.png')
# plt.show()