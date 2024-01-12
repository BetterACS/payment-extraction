import pytest

import sys
import os
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(PATH))

from simple_test.func import compute

TEST_CASES = [
    (1, 1),
    (2, 2),
    (3, 3),
]

@pytest.mark.parametrize("a, expected", TEST_CASES)
def test_compute(a, expected):
    assert compute(a) == expected, f"Input {a} should return {expected} got {compute(a)} instead"
