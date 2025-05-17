from models.openai import OpenAIClient
from models.gemini import GeminiClient
from prompts.prompt import PROMPTS, RAG_PROMPTS
from models.rag import RAG

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import os
import pandas as pd



# Run Inference on the dataset samples using the OpenAI clients
# and save the results to a new CSV file
def run_open_ai_rag_inference_on_dataset(dataset_path,output_path,model, rag):
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

        # Use the RAG model to retrieve similar diffs and their commit messages
        context = rag.retrieve_similar_context(diff, k=4)

        for prompt_name, template in RAG_PROMPTS.items():
            print(f"\n--- Prompt Style: {prompt_name} ---")

            openai_message = ''
            try:
                openai_message = openai_client.generate_commit_message(diff, template, context=context)
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

        break  # Break after the first prompt style only zero_shot

    # Save results to a new DataFrame
    results_df = pd.DataFrame(results)

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")



# Run Inference on the dataset samples using the OpenAI clients
# and save the results to a new CSV file
def run_gemini_rag_inference_on_dataset(dataset_path,output_path,model,rag):
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

        # Use the RAG model to retrieve similar diffs and their commit messages
        context = rag.retrieve_similar_context(diff, k=4)

        for prompt_name, template in RAG_PROMPTS.items():
            print(f"\n--- Prompt Style: {prompt_name} ---")

            gemini_message = ''
            try:
                gemini_message = gemini_client.generate_commit_message(diff, template, context=context)
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

            break  # Break after the first prompt style only zero_shot

    # Save results to a new DataFrame
    results_df = pd.DataFrame(results)

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    print("Running inference on the dataset...")

    rag = RAG(index_path="faiss_index_java.index", documents="documents_java.pkl")


    dataset_path = 'dataset/samples/java.csv'
    open_ai_model = 'gpt-3.5-turbo'
    output_path =  f"dataset/samples/evaluation_results/rag_java_evaluation_results_{open_ai_model}.csv"

    run_open_ai_rag_inference_on_dataset(dataset_path,output_path,open_ai_model,rag)
    

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



    # ###################################################################################################################
    # ####################################################### Python ####################################################
    # ###################################################################################################################    
    # # Python
    # print("Running inference on the python dataset...")
    

    # # #  Test(1) Python vs OpenAI gpt-3.5-turbo
    # # dataset_path = 'dataset/samples/py.csv'
    # # open_ai_model = 'gpt-3.5-turbo'
    # # output_path =  f"dataset/samples/evaluation_results/py_evaluation_results_{open_ai_model}_zero_shot_only.csv"
    
    # # # Run inference on the dataset
    # # run_open_ai_inference_on_dataset(dataset_path,output_path,open_ai_model)
    
    # #  Test(2) Python vs Gemini 2.0 Flash
    # dataset_path = 'dataset/samples/py.csv'
    # gemini_model = 'gemini-2.0-flash'
    # output_path =  f"dataset/samples/evaluation_results/py_evaluation_results_{gemini_model}_zero_shot_only.csv"

    # # Run inference on the dataset
    # run_gemini_inference_on_dataset(dataset_path,output_path,gemini_model)


    

    # ###################################################################################################################
    # ######################################################### JS ######################################################
    # ###################################################################################################################
    # # JavaScript
    # print("Running inference on the javascript dataset...")
    

    # # #  Test(1) JavaScript vs OpenAI gpt-3.5-turbo
    # # dataset_path = 'dataset/samples/js.csv'
    # # open_ai_model = 'gpt-3.5-turbo'
    # # output_path =  f"dataset/samples/evaluation_results/js_evaluation_results_{open_ai_model}_zero_shot_only.csv"
    
    # # # Run inference on the dataset
    # # run_open_ai_inference_on_dataset(dataset_path,output_path,open_ai_model)
    
    # #  Test(2) JavaScript vs Gemini 2.0 Flash
    # dataset_path = 'dataset/samples/js.csv'
    # gemini_model = 'gemini-2.0-flash'
    # output_path =  f"dataset/samples/evaluation_results/js_evaluation_results_{gemini_model}_zero_shot_only.csv"

    # # Run inference on the dataset
    # run_gemini_inference_on_dataset(dataset_path,output_path,gemini_model)

    

    
    # ###################################################################################################################
    # ######################################################### PHP #####################################################
    # ###################################################################################################################
    # # PHP
    # print("Running inference on the PHP dataset...")
    

    # # #  Test(1) PHP vs OpenAI gpt-3.5-turbo
    # # dataset_path = 'dataset/samples/php.csv'
    # # open_ai_model = 'gpt-3.5-turbo'
    # # output_path =  f"dataset/samples/evaluation_results/php_evaluation_results_{open_ai_model}_zero_shot_only.csv"
    
    # # # Run inference on the dataset
    # # run_open_ai_inference_on_dataset(dataset_path,output_path,open_ai_model)
    
    # #  Test(2) PHP vs Gemini 2.0 Flash
    # dataset_path = 'dataset/samples/php.csv'
    # gemini_model = 'gemini-2.0-flash'
    # output_path =  f"dataset/samples/evaluation_results/php_evaluation_results_{gemini_model}_zero_shot_only.csv"

    # # Run inference on the dataset
    # run_gemini_inference_on_dataset(dataset_path,output_path,gemini_model)
