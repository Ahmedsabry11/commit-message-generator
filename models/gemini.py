from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    def __init__(self, model="gemini-2.0-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.model = model
    
    def generate_commit_message(self,diff, prompt_template):
        prompt = prompt_template.format(diff=diff)
        # Create a chat completion request
        response = self.client.chat.completions.create(
            model=self.model,
            n=1,
            messages=[
                {"role": "system", "content": "You are commit message generator."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Check for errors in the response
        if 'error' in response:
            raise Exception(f"Error from OpenAI API: {response['error']}")
        
        # print all keys in the response and corresponding values
        for key, value in response.items():
            print(f"{key}: {value}")

        # Check if 'choices' is in the response and has at least one choice
        if 'choices' not in response or len(response['choices']) == 0:
            raise Exception("No choices returned from OpenAI API.")
        
        # Check if 'message' is in the first choice and has 'content'
        if 'message' not in response['choices'][0] or 'content' not in response['choices'][0]['message']:
            raise Exception("No message content returned from OpenAI API.")

        # Return the content of the first choice's message, stripped of leading/trailing whitespace
        # and ensure it's a string
        print(f"Response: {response['choices'][0]['message']['content']}")
        
        return response['choices'][0]['message']['content'].strip()