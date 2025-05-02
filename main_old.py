from models.openai import OpenAIClient
from models.gemini import GeminiClient
from prompts.prompt import PROMPTS
from dataset.samples import GIT_DIFF
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import os
import pandas as pd


# Run Inference on the dataset samples using the OpenAI and Gemini clients
# and save the results to a new CSV file
def run_inference_on_dataset(dataset_path):
    # Initialize the OpenAI and Gemini clients
    openai_client = OpenAIClient()
    gemini_client = GeminiClient()

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Create a list to hold new rows
    results = []

    for index, row in df.iterrows():
        diff = row['diff']
        expected_message = row['message']

        for prompt_name, template in PROMPTS.items():
            print(f"\n--- Prompt Style: {prompt_name} ---")

            openai_message = ''
            gemini_message = ''

            print("\n--- OpenAI ---")
            try:
                openai_message = openai_client.generate_commit_message(diff, template)
                print("OpenAI message:", openai_message)
            except Exception as e:
                print("OpenAI client failed:", e)

            print("\n--- Gemini ---")
            try:
                gemini_message = gemini_client.generate_commit_message(diff, template)
                print("Gemini message:", gemini_message)
            except Exception as e:
                print("Gemini client failed:", e)

            # Append a new row for this prompt style
            results.append({
                'diff': diff,
                'expected_message': expected_message,
                'prompt_style': prompt_name,
                'openai_message': openai_message,
                'gemini_message': gemini_message
            })

            break

    # Save results to a new DataFrame
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(dataset_path), 'evaluation_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# def evaluate(dataset_path):
#     openai_client = OpenAIClient()
#     gemini_client = GeminiClient()

#     # Load the dataset
#     df = pd.read_csv(dataset_path)

#     # columns is ['diff', 'message']
#     # Iterate over each row in the dataset

#     # create a new column in the dataframe to store the results
#     df['openai_message'] = ''
#     df['gemini_message'] = ''
#     df['expected_message'] = ''
#     df['prompt_style'] = ''


#     for index, row in df.iterrows():
#         diff = row['diff']
#         expected_message = row['message']

#         # Iterate over each prompt style
#         for name, template in PROMPTS.items():
#             print(f"\n--- Prompt Style: {name} ---")
#             print("\n--- Openai ---")
#             try:
#                 openai_message = openai_client.generate_commit_message(diff, template)
#                 print("commit message = ", openai_message)
#                 print("expected message = ", expected_message)
#             except Exception as e:
#                 print("OpenAI client failed to generate commit message:", e)

#             print("\n--- Gemini ---")
#             try:
#                 gemini_message = gemini_client.generate_commit_message(diff, template)
#                 print("commit message = ", gemini_message)
#                 print("expected message = ", expected_message)
#             except Exception as e:
#                 print("Gemini client failed to generate commit message:", e)
            
#             # save the result
#             df.at[index, 'openai_message'] = openai_message
#             df.at[index, 'gemini_message'] = gemini_message
#             df.at[index, 'expected_message'] = expected_message
#             df.at[index, 'prompt_style'] = name
        
#     # Save the results to a new CSV file
#     output_path = os.path.join(os.path.dirname(dataset_path), 'evaluation_results.csv')
#     df.to_csv(output_path, index=False)

#     print(f"Evaluation results saved to {output_path}")


# if __name__ == "__main__":
    
#     # Initialize the OpenAI and Gemini clients
#     openai_client = OpenAIClient()
#     gemini_client = GeminiClient()

#     for name, template in PROMPTS.items():
#         print(f"\n--- Prompt Style: {name} ---")
#         # TODO: filter GIT_DIFF ti remove any lines that are not relevant to the commit message
#         print("\n--- Openai ---")
#         try:
#             message = openai_client.generate_commit_message(GIT_DIFF, template)
#             print("commit message = ",message)
            
#         except Exception as e:
#             print("Gemini client failed to generate commit message:", e)

#         print("\n--- Gemini ---")
#         try:
#             message = gemini_client.generate_commit_message(GIT_DIFF, template)
#             print("--- commit message using Gemini ---")
#             print("commit message = ",message)
#         except Exception as e:
#             print("Gemini client failed to generate commit message:", e)
            

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = 'dataset/samples/java.csv'
    
    # Evaluate the dataset
    run_inference_on_dataset(dataset_path)