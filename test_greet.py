"""Unit tests for the greet utility function."""

import unittest

from greet import greet


class TestGreet(unittest.TestCase):
    """Tests for greet()."""

    def test_greet_regular_name(self) -> None:
        self.assertEqual(greet("Alice"), "Hello, Alice!")

    def test_greet_another_name(self) -> None:
        self.assertEqual(greet("Bob"), "Hello, Bob!")

    def test_greet_empty_string(self) -> None:
        self.assertEqual(greet(""), "Hello, !")

    def test_greet_name_with_spaces(self) -> None:
        self.assertEqual(greet("Jane Doe"), "Hello, Jane Doe!")


if __name__ == "__main__":
    unittest.main()
