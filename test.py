import pickle
import faiss
from models.openai import OpenAIClient
from models.gemini import GeminiClient
from models.rag import RAG
from prompts.prompt import PROMPTS

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import os
import pandas as pd



# Run Inference on the dataset samples using the OpenAI clients
# and save the results to a new CSV file
def run_open_ai_inference_on_dataset(dataset_path,output_path,model, prompt_style, row_index):
    # Initialize the OpenAI clients
    openai_client = OpenAIClient(model=model)

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # get the diff and expected message for the specified row index
    diff = df.loc[row_index, 'diff']
    expected_message = df.loc[row_index, 'message']
    print(f"Processing row {row_index + 1}/{len(df)}")
    print(f"Diff: {diff}\n\n")
    print(f"Expected message: {expected_message}")
    
    # run the OpenAI client with the specified prompt style
    openai_message = ''
    try:
        openai_message = openai_client.generate_commit_message(diff, PROMPTS[prompt_style])
        print("OpenAI message:", openai_message)
    except Exception as e:
        print("OpenAI client failed:", e)
    
    # save the results in a file
    results = {
        'diff': diff,
        'expected_message': expected_message,
        'prompt_style': prompt_style,
        'inference_message': openai_message,
    }

    # Save results to a new DataFrame
    results_df = pd.DataFrame([results])
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")






# Run Inference on the dataset samples using the OpenAI clients
# and save the results to a new CSV file
def run_gemini_inference_on_dataset(dataset_path,output_path,model, prompt_style, row_index):
    # Initialize the Gemini clients
    gemini_client = GeminiClient(model=model)

    # Load the dataset
    df = pd.read_csv(dataset_path)


    # get the diff and expected message for the specified row index
    diff = df.loc[row_index, 'diff']
    expected_message = df.loc[row_index, 'message']
    print(f"Processing row {row_index + 1}/{len(df)}")
    # print(f"Diff: {diff}")
    print(f"Expected message: {expected_message}")
    
    # run the OpenAI client with the specified prompt style
    gemini_message = ''
    try:
        gemini_message = gemini_client.generate_commit_message(diff, PROMPTS[prompt_style])
        print("Gemini message:", gemini_message)
    except Exception as e:
        print("Gemini client failed:", e)
    
    # save the results in a file
    results = {
        'diff': diff,
        'expected_message': expected_message,
        'prompt_style': prompt_style,
        'inference_message': gemini_message,
    }

    # Save results to a new DataFrame
    results_df = pd.DataFrame([results])
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    
    # Load the index and documents

    index = faiss.read_index("faiss_index_java.index")
    with open("documents_java.pkl", "rb") as f:
        documents = pickle.load(f)

    # Initialize RAG
    rag = RAG(index_path="faiss_index_java.index", documents=documents)

    # Example usage
    diff = '''diff --git a/support/cas-server-support-throttle-bucket4j/src/main/java/org/apereo/cas/web/Bucket4jThrottledRequestExecutor.java b/support/cas-server-support-throttle-bucket4j/src/main/java/org/apereo/cas/web/Bucket4jThrottledRequestExecutor.java
index <HASH>..<HASH> 100644
--- a/support/cas-server-support-throttle-bucket4j/src/main/java/org/apereo/cas/web/Bucket4jThrottledRequestExecutor.java
+++ b/support/cas-server-support-throttle-bucket4j/src/main/java/org/apereo/cas/web/Bucket4jThrottledRequestExecutor.java
@@ -53,8 +53,9 @@ public class Bucket4jThrottledRequestExecutor implements ThrottledRequestExecuto
             if (this.blocking) {
                 LOGGER.trace(""Attempting to consume a token for the authentication attempt"");
                 result = !this.bucket.tryConsume(1, MAX_WAIT_NANOS, BlockingStrategy.PARKING);
+            } else {
+                result = !this.bucket.tryConsume(1);
             }
-            result = !this.bucket.tryConsume(1);
         } catch (final InterruptedException e) {
             LOGGER.error(e.getMessage(), e);
             Thread.currentThread().interrupt();'''
    context = rag.retrieve_similar_context(diff, k=4, ignore_first=True)
    print("Retrieved context:")
    print(context)
    # print("Running inference on the dataset...")

    # row_index = 2
    # # Run inference on the dataset using the OpenAI client
    # output_path = f"dataset/samples/evaluation_results/openai_results_js_{row_index}.csv"
    # run_open_ai_inference_on_dataset(
    #     dataset_path="dataset/samples/js.csv",
    #     output_path=output_path,
    #     model="gpt-4-turbo",
    #     prompt_style="feature",
    #     row_index=row_index
    # )

    # # Run inference on the dataset using the Gemini client
    # output_path = f"dataset/samples/evaluation_results/gemini_results_js_{row_index}.csv"
    # run_gemini_inference_on_dataset(
    #     dataset_path="dataset/samples/js.csv",
    #     output_path=output_path,
    #     model="gemini-2.0-flash",
    #     prompt_style="feature",
    #     row_index=row_index
    # )
    