from models.openai import OpenAIClient
from models.gemini import GeminiClient
from prompts.prompt import PROMPTS
from dataset.samples import GIT_DIFF
import os
from dotenv import load_dotenv

load_dotenv()
if __name__ == "__main__":
    
    # Initialize the OpenAI and Gemini clients
    openai_client = OpenAIClient()
    gemini_client = GeminiClient()

    for name, template in PROMPTS.items():
        print(f"\n--- Prompt Style: {name} ---")
        # TODO: filter GIT_DIFF ti remove any lines that are not relevant to the commit message
        print("\n--- Openai ---")
        try:
            message = openai_client.generate_commit_message(GIT_DIFF, template)
            print("commit message = ",message)
            
        except Exception as e:
            print("Gemini client failed to generate commit message:", e)

        print("\n--- Gemini ---")
        try:
            message = gemini_client.generate_commit_message(GIT_DIFF, template)
            print("--- commit message using Gemini ---")
            print("commit message = ",message)
        except Exception as e:
            print("Gemini client failed to generate commit message:", e)
            

