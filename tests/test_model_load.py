import pytest

import sys

sys.path.append("..")
from utils.load_model import load_model


def test_load_model():
    reader, model, processor, device = load_model()
    assert reader is not None
    assert model is not None
    assert processor is not None
    assert device is not None
    assert device == "cuda" or device == "cpu"
