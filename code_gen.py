import os
from dotenv import load_dotenv

import openai

load_dotenv()  # take environment variables from .env.

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_code(prompt, temperature=0, max_tokens=64, frequency_penalty=0, best_of=1):
    prompt = f'"""{prompt}"""'
    result = openai.Completion.create(
        engine="code-davinci-001",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        best_of=best_of
        )
    code = result["choices"][0]["text"].split("\n")
    code = list(filter(lambda x: x != '', code))
    return code