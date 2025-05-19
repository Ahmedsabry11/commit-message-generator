from models.openai import OpenAIClient
from models.gemini import GeminiClient
from prompts.prompt import PROMPTS

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import os
import pandas as pd

import time



# Run Inference on the dataset samples using the OpenAI clients
# and save the results to a new CSV file
def run_open_ai_inference_on_dataset(dataset_path,output_path,model,prompts_list):
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
            # Skip the prompt styles that are not in the list
            if prompt_name not in prompts_list:
                continue

            print(f"\n--- Prompt Style: {prompt_name} ---")

            openai_message = ''

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

            time.sleep(4.5)  # ~13 requests per minute (safer than 15)
            # break  # Break after the first prompt style only zero_shot

    # Save results to a new DataFrame
    results_df = pd.DataFrame(results)

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")



# Run Inference on the dataset samples using the OpenAI clients
# and save the results to a new CSV file
def run_gemini_inference_on_dataset(dataset_path,output_path,model,prompts_list):
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
            # Skip the prompt styles that are not in the list
            if prompt_name not in prompts_list:
                continue

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

            # After each API call
            time.sleep(4.5)  # ~13 requests per minute (safer than 15)

            # break  # Break after the first prompt style only zero_shot

    # Save results to a new DataFrame
    results_df = pd.DataFrame(results)

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    print("Running inference on the dataset...")

    # ###################################################################################################################
    # ######################################################## Java #####################################################
    # ###################################################################################################################
    # # Java
    # print("Running inference on the java dataset...")
    

    # #  Test(1) Java vs OpenAI gpt-3.5-turbo
    # dataset_path = 'dataset/samples/java.csv'
    # # open_ai_model = 'gpt-3.5-turbo'
    # # open_ai_model = 'gpt-4.1-mini'
    # # open_ai_model = 'gpt-4-turbo'
    # open_ai_model = 'gpt-4o'
    # prompt_name = 'feature'
    # # prompt_name = 'zero_shot'
    # output_path =  f"dataset/samples/evaluation_results/java_evaluation_results_{open_ai_model}_{prompt_name}.csv"
    
    # # Run inference on the dataset
    # run_open_ai_inference_on_dataset(dataset_path,output_path,open_ai_model,prompts_list=[prompt_name])
    
    # #  Test(2) Java vs Gemini 2.0 Flash
    # dataset_path = 'dataset/samples/java.csv'
    # gemini_model = 'gemini-2.0-flash'
    # # prompt_name = 'feature'
    # prompt_name = 'zero_shot'
    # output_path =  f"dataset/samples/evaluation_results/java_evaluation_results_{gemini_model}_{prompt_name}.csv"

    # # Run inference on the dataset
    # run_gemini_inference_on_dataset(dataset_path,output_path,gemini_model,prompts_list=[prompt_name])



    # ###################################################################################################################
    # ####################################################### Python ####################################################
    # ##################################################################################################################    
    # # Python
    # print("Running inference on the python dataset...")
    

    # #  Test(1) Python vs OpenAI gpt-3.5-turbo
    # dataset_path = 'dataset/samples/py.csv'
    # # open_ai_model = 'gpt-3.5-turbo'
    # # open_ai_model = 'gpt-4.1-mini'
    # # open_ai_model = 'gpt-4-turbo'
    # open_ai_model = 'gpt-4o'
    # prompt_name = 'feature'
    # # prompt_name = 'zero_shot'
    # output_path =  f"dataset/samples/evaluation_results/py_evaluation_results_{open_ai_model}_{prompt_name}.csv"
    
    # # Run inference on the dataset
    # run_open_ai_inference_on_dataset(dataset_path,output_path,open_ai_model,prompts_list=[prompt_name])
    
    # #  Test(2) Python vs Gemini 2.0 Flash
    # dataset_path = 'dataset/samples/py.csv'
    # gemini_model = 'gemini-2.0-flash'
    # prompt_name = 'feature'
    # output_path =  f"dataset/samples/evaluation_results/py_evaluation_results_{gemini_model}_{prompt_name}.csv"

    # # Run inference on the dataset
    # run_gemini_inference_on_dataset(dataset_path,output_path,gemini_model,prompts_list=[prompt_name])



    # ###################################################################################################################
    # ######################################################### JS ######################################################
    # ###################################################################################################################
    # # JavaScript
    # print("Running inference on the javascript dataset...")
    

    # #  Test(1) JavaScript vs OpenAI gpt-3.5-turbo
    # dataset_path = 'dataset/samples/js.csv'
    # # open_ai_model = 'gpt-3.5-turbo'
    # # open_ai_model = 'gpt-4.1-mini'
    # # open_ai_model = 'gpt-4-turbo'
    # open_ai_model = 'gpt-4o'
    # prompt_name = 'feature'
    # # prompt_name = 'zero_shot'
    # output_path =  f"dataset/samples/evaluation_results/js_evaluation_results_{open_ai_model}_{prompt_name}.csv"
    
    # # Run inference on the dataset
    # run_open_ai_inference_on_dataset(dataset_path,output_path,open_ai_model,prompts_list=[prompt_name])
    
    # #  Test(2) JavaScript vs Gemini 2.0 Flash
    # dataset_path = 'dataset/samples/js.csv'
    # gemini_model = 'gemini-2.0-flash'
    # prompt_name = 'feature'
    # output_path =  f"dataset/samples/evaluation_results/js_evaluation_results_{gemini_model}_{prompt_name}.csv"

    # # Run inference on the dataset
    # run_gemini_inference_on_dataset(dataset_path,output_path,gemini_model,prompts_list=[prompt_name])

    
    ###################################################################################################################
    ######################################################### PHP #####################################################
    ###################################################################################################################
    # PHP
    print("Running inference on the PHP dataset...")
    

    #  Test(1) PHP vs OpenAI gpt-3.5-turbo
    dataset_path = 'dataset/samples/php.csv'
    # open_ai_model = 'gpt-3.5-turbo'
    # open_ai_model = 'gpt-4.1-mini'
    # open_ai_model = 'gpt-4-turbo'
    open_ai_model = 'gpt-4o'
    prompt_name = 'feature'
    # prompt_name = 'zero_shot'
    output_path =  f"dataset/samples/evaluation_results/php_evaluation_results_{open_ai_model}_{prompt_name}.csv"
    
    # Run inference on the dataset
    run_open_ai_inference_on_dataset(dataset_path,output_path,open_ai_model,prompts_list=[prompt_name])
    
    # #  Test(2) PHP vs Gemini 2.0 Flash
    # dataset_path = 'dataset/samples/php.csv'
    # gemini_model = 'gemini-2.0-flash'
    # prompt_name = 'feature'
    # output_path =  f"dataset/samples/evaluation_results/php_evaluation_results_{gemini_model}_{prompt_name}.csv"

    # # Run inference on the dataset
    # run_gemini_inference_on_dataset(dataset_path,output_path,gemini_model,prompts_list=[prompt_name])
