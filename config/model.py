TROCR_MODEL = "betteracs/trocr-th-finetune-receipt-v2"
DETECTOR_JOIN_THRESHOLD = (0.03, 0.03)
IMAGE_SAMPLE = "previews/testimage.jpg"

# region LLM
SYSTEM_PROMPT = (
    """
    You're Thai document AI. 
    - You can turn unstructured data into a structure data with your knowledge.
    - You understand Thai OCR result as well. (So you know how to handle OCR result)
    - You can help to convert text to JSON format.
    - Although you're smart, you won't explain what have you done. (output what user need)
    """,
)
MAX_TOKENS = 300
TEMPERATURE = 0.6
TOP_P = 1
# endregion
