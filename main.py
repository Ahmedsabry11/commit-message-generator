from models.openai import OpenAIClient
from models.gemini import GeminiClient
from prompts.prompt import PROMPTS

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import os
import pandas as pd



# Run Inference on the dataset samples using the OpenAI clients
# and save the results to a new CSV file
def run_open_ai_inference_on_dataset(dataset_path,output_path,model):
    # Initialize the OpenAI clients
    openai_client = OpenAIClient(model=model)

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Create a list to hold new rows
    results = []

    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{len(df)}")

        diff = row['diff']
        expected_message = row['message']

        for prompt_name, template in PROMPTS.items():
            print(f"\n--- Prompt Style: {prompt_name} ---")

            openai_message = ''
            gemini_message = ''

            try:
                openai_message = openai_client.generate_commit_message(diff, template)
                print("OpenAI message:", openai_message)
            except Exception as e:
                print("OpenAI client failed:", e)

                    # Append a new row for this prompt style
            results.append({
                'diff': diff,
                'expected_message': expected_message,
                'prompt_style': prompt_name,
                'inference_message': openai_message,
            })

    # Save results to a new DataFrame
    results_df = pd.DataFrame(results)

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")



# Run Inference on the dataset samples using the OpenAI clients
# and save the results to a new CSV file
def run_gemini_inference_on_dataset(dataset_path,output_path,model):
    # Initialize the Gemini clients
    gemini_client = GeminiClient(model=model)

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Create a list to hold new rows
    results = []

    for index, row in df.iterrows():
        print("=" * 100)
        print(f"Processing row {index + 1}/{len(df)}")

        diff = row['diff']
        expected_message = row['message']

        for prompt_name, template in PROMPTS.items():
            print(f"\n--- Prompt Style: {prompt_name} ---")

            gemini_message = ''

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
                'inference_message': gemini_message,
            })

    # Save results to a new DataFrame
    results_df = pd.DataFrame(results)

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # ###################################################################################################################
    # ######################################################## Java #####################################################
    # ###################################################################################################################
    # # Java
    # print("Running inference on the java dataset...")
    

    # #  Test(1) Java vs OpenAI gpt-3.5-turbo
    # dataset_path = 'dataset/samples/java.csv'
    # open_ai_model = 'gpt-3.5-turbo'
    # output_path =  f"dataset/samples/evaluation_results/java_evaluation_results_{open_ai_model}.csv"
    
    # # Run inference on the dataset
    # run_open_ai_inference_on_dataset(dataset_path,output_path,open_ai_model)
    
    # #  Test(2) Java vs Gemini 2.0 Flash
    # dataset_path = 'dataset/samples/java.csv'
    # gemini_model = 'gemini-2.0-flash'
    # output_path =  f"dataset/samples/evaluation_results/java_evaluation_results_{gemini_model}.csv"

    # # Run inference on the dataset
    # run_gemini_inference_on_dataset(dataset_path,output_path,gemini_model)



    ###################################################################################################################
    ######################################################## Java #####################################################
    ###################################################################################################################
    # Java
    print("Running inference on the java dataset...")
    

    #  Test(1) Java vs OpenAI gpt-3.5-turbo
    dataset_path = 'dataset/samples/java.csv'
    open_ai_model = 'gpt-3.5-turbo'
    output_path =  f"dataset/samples/evaluation_results/java_evaluation_results_{open_ai_model}.csv"
    
    # Run inference on the dataset
    run_open_ai_inference_on_dataset(dataset_path,output_path,open_ai_model)
    
    #  Test(2) Java vs Gemini 2.0 Flash
    dataset_path = 'dataset/samples/java.csv'
    gemini_model = 'gemini-2.0-flash'
    output_path =  f"dataset/samples/evaluation_results/java_evaluation_results_{gemini_model}.csv"

    # Run inference on the dataset
    run_gemini_inference_on_dataset(dataset_path,output_path,gemini_model)