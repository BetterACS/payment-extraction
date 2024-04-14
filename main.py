from typing import Any, Dict

import uvicorn
from fastapi import FastAPI

from utils import load_model, download_image
from utils.detect import detect_pipeline

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/receipt/extract/")
async def extract_text_from_image(url: str = "") -> Dict[str, Any]:
    if url == "":
        return {"code": 400, "message": "No image URL provided"}

    name = download_image(url)
    if name == "":
        return {"code": 400, "message": "Invalid image URL"}

    reader, model, processor, device = load_model()
    images_list, dataframe = detect_pipeline(reader, name)
    if len(images_list) == 0:
        return {"code": 400, "message": "No text detected"}

    texts = []

    for img in images_list:
        pixels_value = processor(img, return_tensors="pt").pixel_values
        outputs = model.generate(pixels_value.to("cuda"))
        generated_text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        texts.append(generated_text)

    dataframe["text"] = texts

    return {"code": 200, "message": "Text extracted successfully", "data": dataframe["text"].tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
