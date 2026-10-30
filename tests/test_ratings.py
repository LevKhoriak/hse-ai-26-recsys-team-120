"""Тесты пакета mm_recsys."""

import mm_recsys
from mm_recsys.ratings import mean_rating


def test_package_imports() -> None:
    assert mm_recsys.__doc__ is not None


def test_mean_rating() -> None:
    assert mean_rating([5.0, 4.0, 3.0]) == 4.0


def test_mean_rating_empty() -> None:
    assert mean_rating([]) == 0.0
