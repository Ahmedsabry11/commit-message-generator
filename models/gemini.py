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

        try:
            # # Mocked response for testing purposes
            # return "Gemini client is not available."
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
            
            # Check if 'choices' is in the response and has at least one choice
            if response.choices is None or len(response.choices) == 0:
                raise Exception("No choices returned from OpenAI API.")
            
            # Check if 'message' is in the first choice and has 'content'
            if response.choices[0].message is None or response.choices[0].message.content is None:
                raise Exception("No message content returned from OpenAI API.")

            return response.choices[0].message.content.strip()
        
        except Exception as e:
            raise Exception(f"Failed to generate commit message: {e}")