import os, sys
sys.path.append("Grounded-Segment-Anything")
import argparse
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import matplotlib.pyplot as plt

import PIL
from huggingface_hub import hf_hub_download


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_segment_corners(mask):
    # Ensure the input is a NumPy array of type uint8
    mask = np.array(mask, dtype=np.uint8)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # No segment found
    # Assuming the largest contour is the segment of interest
    contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon with less vertices
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # print(approx)
    
    # Find the four corners
    if len(approx) >= 4:
        corners = [(point[0][0], point[0][1]) for point in approx]
        
        topleft_idx = find_nearest_corner(corners, (0, 0)).argmin()
        topright_idx = find_nearest_corner(corners, (mask.shape[1], 0)).argmin()
        bottomleft_idx = find_nearest_corner(corners, (0, mask.shape[0])).argmin()
        bottomright_idx = find_nearest_corner(corners, (mask.shape[1], mask.shape[0])).argmin()

        corners = [
            corners[topleft_idx], 
            corners[topright_idx], 
            corners[bottomleft_idx], 
            corners[bottomright_idx]
        ]
    else:
        # In case the approximation doesn't return exactly 4 points, 
        # we'll fall back to finding the bounding box corners.
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
    
    return corners

def find_nearest_corner(corners, target):
    distance_list = []
    
    for corner in corners:
        distance_list.append(euclidean_distance(corner, target))
    
    return np.array(distance_list)
    
def euclidean_distance(vector1, vector2):
    # Convert inputs to NumPy arrays if they aren't already
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    # Calculate the Euclidean distance
    distance = np.linalg.norm(vector1 - vector2)
    return distance

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def get_receipt_only(path, groundingdino_model, sam_predictor):
    TEXT_PROMPT = "receipt"
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    # local_image_path = "logs/images.jpg"

    image_source, image = load_image(path)

    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD,
        device=DEVICE
    )

    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    )
    masks = masks[0][0].cpu()
    cv2.imwrite("segment_image.jpg", show_mask(masks, image_source))
    
    corners = find_segment_corners(masks)
    desired_points = np.float32(((0, 0), (524, 0), (0, 960), (524, 960)))
    corner = np.float32(corners)

    matrix = cv2.getPerspectiveTransform(corner, desired_points)
    warp_image = cv2.warpPerspective(image_source, matrix, (524, 960))
    
    return warp_image