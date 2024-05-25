import os
import sys
from typing import Dict

import dotenv
from openai import OpenAI
import openai
import json
from pydantic import BaseModel, constr
from groq import Groq
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config.model

dotenv.load_dotenv()


def convert_text_to_json(text: str) -> Dict:
    client = OpenAI(api_key=os.getenv("GPT_API_KEY"))
    chat = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": config.model.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": config.model.INSTRUCTION_PROMPT.replace("{plain_text}", text),
            },
        ],
        seed=888,
        stream=False,
    )

    try:
        output = chat.choices[0].message.content.replace("```json\n", "").replace("`", "")
        output = re.search("\{(.|\n)*\}", output).group()
        output = json.loads(output)
    except Exception as e:
        print(e)
        output = {}

    return output
