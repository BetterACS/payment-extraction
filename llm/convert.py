import os
import sys
from typing import Dict

import dotenv
from openai import OpenAI
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config.model

dotenv.load_dotenv()


def convert_text_to_json(text: str) -> Dict:
    client = OpenAI(
        api_key=os.getenv("TYPHOON_API_KEY"),
        base_url="https://api.opentyphoon.ai/v1",
    )
    chat = client.chat.completions.create(
        model="typhoon-instruct",
        messages=[
            {"role": "system", "content": config.model.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""ช่วยเปลี่ยนเป็น JSON 
                ```
                {text}
                ```
                โดยเป็นลักษณะนี้ 
                ```
                "items": [
                    { 
                        "name": "...", 
                        "price": ..., 
                        "qty": ..., 
                        "discount": .., 
                    }, 
                    ... 
                ],
                "vat": .. (0 if not found)
                ```
                ข้อระวัง: ข้อความอาจมีการสลับตำแหน่งกันได้ โปรดใช้ความสามารถในการหาคำตอบ""",
            },
        ],
        max_tokens=config.model.MAX_TOKENS,
        temperature=config.model.TEMPERATURE,
        top_p=config.model.TOP_P,
        stream=False,
    )

    try:
        output = chat.choices[0].message.content
        output = json.loads(output)
    except Exception as e:
        print(e)
        output = {}

    return output
