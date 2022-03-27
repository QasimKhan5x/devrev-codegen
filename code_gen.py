import os

import openai
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

openai.api_key = os.environ.get("OPENAI_API_KEY")


def get_code(prompt, get_fn_reason=False, **kwargs):
    '''
    Useful parameters
        - temperature/top_p
        - best_of
        - max_tokens
        - frequency_penalty
        - presence_penalty
        - stop
        - n
    '''
    default_kwargs = {
        "top_p": 1.0,
        "max_tokens": 256,
    }
    # set default value if not present in kwargs
    for k, v in default_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v
    result = openai.Completion.create(
        engine="code-davinci-001",
        prompt=prompt,
        **kwargs
    )
    if 'n' in kwargs:
        codes = []
        # [(code, finish_reason)]
        for item in result["choices"]:
            finish_reason = item['finish_reason']
            code = item["text"]
            if get_fn_reason:
                codes.append((code, finish_reason))
            else:
                codes.append(code)
        return codes
    else:
        finish_reason = result['choices'][0]['finish_reason']
        code = result["choices"][0]["text"]
        if get_fn_reason:
            return code, finish_reason
        else:
            return code
