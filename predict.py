# https://f.ptcdn.info/617/045/000/ocry4h3xsDvXLhR6EmN-o.jpg

from typing import Any, Dict

import uvicorn
import cv2
from utils import load_model, download_image, generate_line_list
from utils.detect import detect_pipeline

from llm.convert import convert_text_to_json


if __name__ == "__main__":
    name = "/home/monshinawatra/works/payment-extraction/query_image_20240415230029.jpg"
    reader, model, processor, device = load_model()
    images_list, dataframe = detect_pipeline(reader, name)

    image = cv2.imread(name)

    for idx, row in dataframe.iterrows():
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # texts = []

    # for img in images_list:
    #     pixels_value = processor(img, return_tensors="pt").pixel_values
    #     outputs = model.generate(pixels_value.to("cuda"))
    #     generated_text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    #     texts.append(generated_text)

    # dataframe["text"] = texts

    line_list = generate_line_list(dataframe)
    plain_text = "\n".join(line_list)

    with open("output.txt", "w") as f:
        f.write(plain_text)
