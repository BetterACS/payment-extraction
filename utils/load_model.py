from easyocr import Reader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from functools import lru_cache
import torch
import os, sys
from pathlib import Path

sys.path.append(str(Path(".").absolute() / "Grounded-Segment-Anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything import build_sam, SamPredictor 

sys.path.append("..")
import config.model
from huggingface_hub import hf_hub_download

@lru_cache(maxsize=1)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionEncoderDecoderModel.from_pretrained(config.model.TROCR_MODEL)
    processor = TrOCRProcessor.from_pretrained(config.model.TROCR_MODEL)
    model.to(device)
    reader = Reader(["en", "th"], gpu=False, recognizer=False)

    return reader, model, processor, device

@lru_cache(maxsize=1)
def load_grounded_models():
    repo_id = config.model.ckpt_repo_id
    filename = config.model.ckpt_filenmae
    ckpt_config_filename = config.model.ckpt_config_filename
    device = "cuda"
    
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    
    model.eval()
    
    sam_checkpoint = config.model.sam_checkpoint
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    return model, sam_predictor 