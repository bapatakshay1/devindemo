"""Unit tests for the greet utility function."""

import unittest

from greet import greet


class TestGreet(unittest.TestCase):
    """Tests for greet()."""

    def test_greet_returns_hello_message(self) -> None:
        self.assertEqual(greet("Alice"), "Hello, Alice!")

    def test_greet_with_different_name(self) -> None:
        self.assertEqual(greet("Bob"), "Hello, Bob!")

    def test_greet_with_empty_string(self) -> None:
        self.assertEqual(greet(""), "Hello, !")


if __name__ == "__main__":
    unittest.main()
