import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_commit_message(diff, prompt_template, model="gpt-3.5-turbo"):
    prompt = prompt_template.format(diff=diff)
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.3
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
