import pytest

def test_valid():
    assert 1 == 1, "This should pass"

def test_invalid():
    assert 1 == 2, "This should fail"