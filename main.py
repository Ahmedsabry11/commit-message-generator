from models.openai import generate_commit_message
from prompts.prompt import PROMPTS
from data.samples import GIT_DIFF

if __name__ == "__main__":
    for name, template in PROMPTS.items():
        print(f"\n--- Prompt Style: {name} ---")
        # TODO: filter GIT_DIFF ti remove any lines that are not relevant to the commit message
        message = generate_commit_message(GIT_DIFF, template)
        print(message)
