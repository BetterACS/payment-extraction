import pytest
import easyocr
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1].absolute()))
from utils.load_model import load_model
from utils.detect import detect_text, merge_bbox, detect_pipeline
from config.model import IMAGE_SAMPLE

MERGE_BBOX_TEST_CASES = [
    # Test case 1
    ([(1, 1, 1, 1), (2, 2, 2, 2), (1000, 1000, 1000, 1000)], 150, 150, [(1, 2, 1, 2), (1000, 1000, 1000, 1000)])
]


@pytest.fixture(scope="module")
def reader() -> easyocr.Reader:
    return easyocr.Reader(["th", "en"], gpu=False, recognizer=False)


def test_load_model():
    reader, model, processor, device = load_model()
    assert reader is not None
    assert model is not None
    assert processor is not None
    assert device is not None
    assert device == "cuda" or device == "cpu"


def test_detect_text(reader: easyocr.Reader):
    points = detect_text(reader, IMAGE_SAMPLE)
    assert len(points) > 0
    for point in points:
        assert len(point) == 4


@pytest.mark.parametrize("points, width, height, expected", MERGE_BBOX_TEST_CASES)
def test_merge_bbox(points, width, height, expected):
    result = merge_bbox(points, width, height)
    count = len(expected)
    assert len(result) == count

    for bbox in result:
        assert bbox in expected
        expected.remove(bbox)

        count -= 1

    assert count == 0


def test_detect_pipeline(reader: easyocr.Reader):
    text_boxes, dataframe = detect_pipeline(reader, IMAGE_SAMPLE)
    assert len(text_boxes) > 0
    assert dataframe is not None
    assert len(dataframe) > 0
    assert len(dataframe.columns) == 4
