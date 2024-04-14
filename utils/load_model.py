from easyocr import Reader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from functools import lru_cache
import torch
import sys

sys.path.append("..")
import config.model


@lru_cache(maxsize=1)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionEncoderDecoderModel.from_pretrained(config.model.TROCR_MODEL)
    processor = TrOCRProcessor.from_pretrained(config.model.TROCR_MODEL)
    model.to(device)
    reader = Reader(["en", "th"], gpu=False, recognizer=False)

    return reader, model, processor, device
