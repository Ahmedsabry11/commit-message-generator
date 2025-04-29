PROMPTS = {
    "zero_shot": "Write a commit message only for this diff in one sentence:\n\n{diff}\n\n",
    "conventional": "Write a Conventional Commit message (e.g., feat:, fix:, add:, update:, remove:, refactor:, etc) for this diff:\n\n{diff} \n\n give just commit message",
    "imperative": "Describe the following change using an imperative tone:\n\n{diff}\n\n give just commit message",
    "minimal": "Generate a minimal commit message for the following diff:\n\n{diff} \n\n give just commit message",
    "detailed": "Provide a detailed commit message for the following diff, explaining the changes made:\n\n{diff} \n\n give just commit message",
}

