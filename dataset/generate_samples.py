import pandas as pd
import random

def pick_random_samples(df,count):
    """
    Pick random samples from the dataframe with specific conditions:
    - Commit message length between 5 and 20 words.
    - Diff length between 300 and 700 characters.
    """
    # Filter commit messages with 5-20 words and diff lengths between 300 and 700 characters
    df_filtered = df[(df['message'].apply(lambda x: 5 <= len(x.split()) <= 20)) & 
                     (df['diff'].apply(lambda x: 300 <= len(x) <= 700))]
    

    # Pick random samples (make sure we don't sample more than the available rows)
    if len(df_filtered) >= count:
        return df_filtered.sample(n=count, random_state=42)  # Fix random seed for reproducibility
    else:
        print(f"Not enough data. Only {len(df_filtered)} rows available.")
        return df_filtered
    
if __name__ == "__main__":
    # Check if the script is being run directly
    print("Generating Samples")

    files = [
        "dataset/java.csv",
        "dataset/js.csv",
        "dataset/php.csv",
        "dataset/py.csv",
    ]


    for file in files:
        # Get File Name
        fileName = file.split("/")[-1]

        print(f'Generating Samples for {fileName}')

        # Example CSV loading for two CSVs
        df = pd.read_csv(file)

        # Pick 50 random samples from each CSV
        samples = pick_random_samples(df, 50)
  
        samples.to_csv(f'dataset/samples/{fileName}', index=False)