import pytest

import sys
import os
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(PATH))

import asd.func

TEST_CASES = [
    (1, 1),
    (2, 2),
    (3, 3),
]

@pytest.mark.parametrize("a, expected", TEST_CASES)
def test_compute(a, expected):
    assert asd.func.compute(a) == expected
