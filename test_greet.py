"""Unit tests for the greet utility function."""

from greet import greet


def test_greet_basic():
    assert greet("Alice") == "Hello, Alice!"


def test_greet_another_name():
    assert greet("Bob") == "Hello, Bob!"


def test_greet_empty_string():
    assert greet("") == "Hello, !"
