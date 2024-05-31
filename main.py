from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
import cv2
from utils import (
    load_grounded_models, 
    get_receipt_only, 
    load_model, 
    download_image, 
    generate_line_list
)
from utils.detect import detect_pipeline

from llm.convert import convert_text_to_json
from llm.test import test_llm
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    grounded_dino, sam_preditor = load_grounded_models()
    image = get_receipt_only(name, grounded_dino, sam_preditor)

    cv2.imwrite("test_image.jpg", image)

    reader, model, processor, device = load_model()
    
    images_list, dataframe = detect_pipeline(reader, image)
    if len(images_list) == 0:
        return {"code": 400, "message": "No text detected"}

    texts = []

    batch = 32
    for chunk in range(0, len(images_list), batch):
        images = images_list[chunk:chunk+batch]
        pixels_value = processor(images, return_tensors="pt", padding=True).pixel_values
        outputs = model.generate(pixels_value.to("cuda"))
        generated_text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        texts.extend(generated_text)
    
    # for img in images_list:
    #     pixels_value = processor(img, return_tensors="pt").pixel_values
    #     outputs = model.generate(pixels_value.to("cuda"))
    #     generated_text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    #     texts.append(generated_text)

    dataframe["text"] = texts
    dataframe.to_csv(name.replace(".jpg", ".csv"), index=False)
    
    line_list = generate_line_list(dataframe)
    plain_text = "\n".join(line_list)
    with open(name.replace(".jpg", ".txt"), 'w', encoding='utf-8') as log_file:
        log_file.write(plain_text)

    output = convert_text_to_json(plain_text)

    return {"code": 200, "message": "Text extracted successfully", "data": output}

@app.get("/test/llm")
async def test(query: str):
    return {"answer", test_llm(query)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
