import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1].absolute()))
from utils.load_model import load_model


def test_load_model():
    reader, model, processor, device = load_model()
    assert reader is not None
    assert model is not None
    assert processor is not None
    assert device is not None
    assert device == "cuda" or device == "cpu"
