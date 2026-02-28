"""Unit tests for the greet utility function."""

from greet import greet


def test_greet_returns_hello_message() -> None:
    assert greet("Alice") == "Hello, Alice!"


def test_greet_with_different_name() -> None:
    assert greet("Bob") == "Hello, Bob!"


def test_greet_with_empty_string() -> None:
    assert greet("") == "Hello, !"
