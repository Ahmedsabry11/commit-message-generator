import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.client = OpenAI()

    def generate_commit_message(self, diff, prompt_template):
        prompt = prompt_template.format(diff=diff)
        try:
            # # Mocked response for testing purposes
            # return "OpenAI client is not available."
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a commit message generator."},
                    {"role": "user", "content": prompt}
                ],
                n=1,
                temperature=0.3
            )

            # # Print the structured response for debugging
            # print("Full response:", response)
            # print("\n\n")
            
            # Extract and return the commit message content
            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"Failed to generate commit message: {e}")
