import urllib.request
from datetime import datetime
from typing import Any, Dict

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image

from utils import load_model
from utils.detect import detect_text

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/receipt/extract/")
async def extract_text_from_image(url: str = "") -> Dict[str, Any]:
    if url == "":
        return {"code": 400, "message": "No image URL provided"}

    date = datetime.now().strftime("%Y%m%d%H%M%S")
    name = "query_image_" + date + ".jpg"

    try:
        urllib.request.urlretrieve(url, name)
    except:
        return {"code": 400, "message": "Invalid image URL"}

    reader, model, processor, device = load_model()
    images_list = detect_text(reader, name)
    if len(images_list) == 0:
        return {"code": 400, "message": "No text detected"}

    texts = []

    for img in images_list:
        pixels_value = processor(img, return_tensors="pt").pixel_values
        outputs = model.generate(pixels_value.to("cuda"))
        generated_text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        texts.append(generated_text)

    return {"code": 200, "message": "Text extracted successfully", "data": texts}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
