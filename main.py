from typing import Any, Dict

import uvicorn
from fastapi import FastAPI

from utils import load_model, download_image, generate_line_list
from utils.detect import detect_pipeline

from llm.convert import convert_text_to_json

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

    line_list = generate_line_list(dataframe)
    plain_text = "\n".join(line_list)

    output = convert_text_to_json(plain_text)

    return {"code": 200, "message": "Text extracted successfully", "data": output}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
