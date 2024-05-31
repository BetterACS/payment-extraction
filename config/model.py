TROCR_MODEL = "betteracs/trocr-th-finetune-receipt-v2"
DETECTOR_JOIN_THRESHOLD = (0.03, 0.03)
IMAGE_SAMPLE = "previews/testimage.jpg"

# region LLM
SYSTEM_PROMPT = """
You're Thai document AI. 
- You can turn unstructured data into a structure data with your knowledge.
- You understand Thai OCR result as well. (So you know how to handle OCR result)
- You can help to convert text to JSON format.
- Although you're smart, you won't explain what have you done. (output what user need)
"""
INSTRUCTION_PROMPT = """
Your task is to reformat from unordered plain-text to structered json text.

Here is a plain-text
```
{plain_text}
```

Instruction:
    - There are manys text that not relate to the menu please ignore them.
    - Don't use another keys except ('items', 'name', 'price', 'qty', 'discount').
    - If you found null value please fill with 0 or 0.0.
    - The plain-text can be unorder. You should use your knowledge to reorder it.
    - Don't explain or giving any text except json output.

Answer format:
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
"""
MAX_TOKENS = 300
TEMPERATURE = 0.6
TOP_P = 1
# endregion

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
sam_checkpoint = 'sam_vit_h_4b8939.pth'