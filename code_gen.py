import os
from dotenv import load_dotenv

import openai

load_dotenv()  # take environment variables from .env.

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_code(prompt, **kwargs):
    '''
    Useful parameters
        - temperature/top_p
        - best_of
        - max_tokens
        - frequency_penalty
        - presence_penalty
    '''
    default_kwargs = {
        "top_p": 1.0,
        "max_tokens": 256,
    }
    # set default value if not present in kwargs
    for k, v in default_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v
    # wrap in docstring
    prompt = f'"""{prompt}"""'
    result = openai.Completion.create(
        engine="code-davinci-001",
        prompt=prompt,
        **kwargs
        )
    code = result["choices"][0]["text"].split("\n")
    code = list(filter(lambda x: x != '', code))
    return code
