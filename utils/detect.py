import cv2
import numpy as np
from itertools import combinations
from PIL import Image


def detect_text(reader, path: str):
    text_boxes = []
    im = cv2.imread(str(path))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    result = reader.detect(path)
    points = []

    for bbox in result[0][0]:
        x1, x2, y1, y2 = bbox
        points.append([x1, x2, y1, y2])

    if len(points) == 0:
        return []

    normalize_points = points / np.array([im.shape[1], im.shape[1], im.shape[0], im.shape[0]])
    x_join_threshold = 0.03
    y_join_threshold = 0.03

    indices = list(range(len(normalize_points)))
    join_candidates = []

    for idx1, idx2 in combinations(indices, 2):
        x1, x2, y1, y2 = normalize_points[idx1]
        x3, x4, y3, y4 = normalize_points[idx2]

        distance1 = abs(x2 - x3)
        distance2 = abs(y1 - y4)
        if distance1 < x_join_threshold and distance2 < y_join_threshold:
            x1 = min(x1, x3)
            x2 = max(x2, x4)
            y1 = min(y1, y3)
            y2 = max(y2, y4)

            join_candidates.append((idx1, idx2))

    new_points = []
    flatten_join_candidates = [idx for idx1, idx2 in join_candidates for idx in [idx1, idx2]]
    for idx in indices:
        if idx not in flatten_join_candidates:
            new_points.append(points[idx])

    for idx1, idx2 in join_candidates:
        x1, x2, y1, y2 = points[idx1]
        x3, x4, y3, y4 = points[idx2]
        x1 = min(x1, x3)
        x2 = max(x2, x4)
        y1 = min(y1, y3)
        y2 = max(y2, y4)

        new_points.append([x1, x2, y1, y2])

    for point in new_points:
        x1, x2, y1, y2 = point
        # cv2.rectangle(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        image = im[y1:y2, x1:x2]
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            continue

        if image.shape[0] > 26:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
            text_boxes.append(image)

    return text_boxes
