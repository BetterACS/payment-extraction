import os
import sys
from typing import Dict

import dotenv
from openai import OpenAI
import openai
import json
from pydantic import BaseModel, constr
import instructor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config.model

dotenv.load_dotenv()


def convert_text_to_json(text: str) -> Dict:
    client = OpenAI(
        api_key=os.getenv("GPT_API_KEY"),
        # base_url="https://api.opentyphoon.ai/v1",
    )
    chat = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": config.model.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""ช่วยเปลี่ยนเป็น JSON 
                ```json
                {text}
                ```
                """
                + """
                โดยเป็นลักษณะนี้ 
                ```json
                {
                    "items": [
                        { 
                            "name": str, 
                            "price": float, 
                            "qty": float, 
                            "discount": float
                        }
                        ...
                    ],
                }
                ```
                กฏการตอบ:   1. ไม่ควรมี key อื่นไม่ว่ากรณีไหนนอกจากที่ระบุไว้ในคำสั่ง
                            2. หากมีค่าเป็น null ให้ใส่เป็น 0 หรือ 0.0 ตามความเหมาะสม

                ข้อระวัง: 1. ข้อความอาจมีการสลับตำแหน่งกันได้ โปรดใช้ความสามารถในการหาคำตอบ
                        2. ไม่ควรมีการอธิบายเพิ่มเติม หรือการคอมเม้นต์
                """,
            },
        ],
        response_format={"type": "json_object"},
        # function_call={"name": "story_meta_data"},
        # functions=[mock_function],
        seed=888,
        # max_tokens=config.model.MAX_TOKENS,
        # temperature=config.model.TEMPERATURE,
        # top_p=config.model.TOP_P,
        stream=False,
    )

    try:
        output = chat.choices[0].message.content.replace("```json\n", "").replace("`", "")
        output = json.loads(output)
    except Exception as e:
        print(e)
        output = {}

    return output
