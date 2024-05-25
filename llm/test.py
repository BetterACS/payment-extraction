import os
import sys
from typing import Dict

import dotenv
from openai import OpenAI
import openai
import json
from pydantic import BaseModel, constr

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config.model

def test_llm(query):
    client = OpenAI(
        api_key=os.getenv("GPT_API_KEY"),
    )
    chat = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            # {"role": "system", "content": config.model.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": query,
            },
        ],
        # response_format={"type": "json_object"},
        # function_call={"name": "story_meta_data"},
        # functions=[mock_function],
        seed=888,
        # max_tokens=config.model.MAX_TOKENS,
        # temperature=config.model.TEMPERATURE,
        # top_p=config.model.TOP_P,
        stream=False,
    )
    
    try:
        output = chat.choices[0].message.content#.replace("```json\n", "").replace("`", "")
        # output = json.loads(output)
    except Exception as e:
        print(e)
        output = {}
        
    print("output", output)

    return output