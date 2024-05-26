import sys
from itertools import combinations
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate

sys.path.append("..")
import config.model

BBOX = Tuple[int, int, int, int]

def detect_text(reader, image) -> List[BBOX]:
    # im = cv2.imread(str(path))
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # angle = get_angle(im)
    # im = rotate(im, angle)
    
    # Detect text from image
    result = reader.detect(image)
    points = [point for point in result[0][0]]

    if len(points) == 0:
        return []

    return points


def merge_bbox(points: List[BBOX], width: int, height: int) -> List[BBOX]:
    """
    Merge bounding boxes that are close to each other.

    Args:
        points: List of bounding boxes.
        width: int, image width.
        height: int, image height.

    Returns:
        List of merged bounding boxes.
    """
    join_x_threshold, join_y_threshold = config.model.DETECTOR_JOIN_THRESHOLD
    # Normalize points
    normalize_points = points / np.array([width, width, height, height])
    indices = list(range(len(normalize_points)))

    join_candidates = []

    for idx1, idx2 in combinations(indices, 2):
        x1, x2, y1, y2 = normalize_points[idx1]
        x3, x4, y3, y4 = normalize_points[idx2]

        # Check if the bounding boxes are close to each other
        # From two distances, we can determine if the bounding boxes are close to each other.
        distance1 = abs(x2 - x3)
        distance2 = abs(y1 - y4)
        if distance1 < join_x_threshold and distance2 < join_y_threshold:
            x1 = min(x1, x3)
            x2 = max(x2, x4)
            y1 = min(y1, y3)
            y2 = max(y2, y4)

            # Add the bounding boxes to join_candidates
            join_candidates.append((idx1, idx2))

    # Flatten join_candidates
    flatten_join_candidates = [idx for idx1, idx2 in join_candidates for idx in [idx1, idx2]]
    new_points = []
    # Add the bounding boxes that are not in join_candidates (i.e., not close to each other)
    for idx in indices:
        if idx not in flatten_join_candidates:
            new_points.append(points[idx])

    # Merge the bounding boxes that are close to each other.
    for idx1, idx2 in join_candidates:
        x1, x2, y1, y2 = points[idx1]
        x3, x4, y3, y4 = points[idx2]
        x1 = min(x1, x3)
        x2 = max(x2, x4)
        y1 = min(y1, y3)
        y2 = max(y2, y4)

        new_points.append((x1, x2, y1, y2))

    return new_points


def detect_pipeline(reader, image) -> Tuple[List[Image.Image], pd.DataFrame]:
    # Detect text from image
    points = detect_text(reader, image)
    if len(points) == 0:
        return [], pd.DataFrame()

    im = image
    height, width, _ = im.shape
    # Merge bounding boxes
    new_points = merge_bbox(points, width, height)

    dataframe_dict = {
        "x1": [],
        "x2": [],
        "y1": [],
        "y2": [],
    }
    text_boxes = []

    for point in new_points:
        x1, x2, y1, y2 = point
        image = im[y1:y2, x1:x2]
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            continue

        # Add the image to text_boxes if the height is greater than 20.
        if image.shape[0] < 20:
            continue

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
        text_boxes.append(image)

        dataframe_dict["x1"].append(x1)
        dataframe_dict["x2"].append(x2)
        dataframe_dict["y1"].append(y1)
        dataframe_dict["y2"].append(y2)

    return text_boxes, pd.DataFrame(dataframe_dict)
