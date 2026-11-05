"""Shared fixtures. Nothing in this suite may touch real S3 or Hugging Face."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest

from recsys.data import s3
from recsys.settings import get_settings

BUCKET = "test-bucket"


def _no_network(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("test tried to reach real S3")


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin settings and block real S3 connections for every test."""
    monkeypatch.setenv("S3_BUCKET_NAME", BUCKET)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.test")
    monkeypatch.setenv("AWS_REGION", "test-region")
    monkeypatch.setattr(s3, "_fs", _no_network)
    monkeypatch.setattr(s3, "_client", _no_network)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeS3:
    """In-memory stand-in for the typed s3fs wrappers in recsys.data.s3."""

    def __init__(self) -> None:
        self.objects: set[str] = set()

    @staticmethod
    def _key(path: str) -> str:
        return path.removeprefix("s3://").rstrip("/")

    def add(self, *paths: str) -> None:
        self.objects.update(self._key(p) for p in paths)

    def exists(self, path: str) -> bool:
        key = self._key(path)
        return any(o == key or o.startswith(f"{key}/") for o in self.objects)

    def find(self, path: str) -> list[str]:
        key = self._key(path)
        return sorted(o for o in self.objects if o.startswith(f"{key}/"))

    def ls(self, path: str) -> list[str]:
        key = self._key(path)
        children = {f"{key}/{o[len(key) + 1 :].split('/')[0]}" for o in self.find(key)}
        return sorted(children)

    def copy(self, src: str, dst: str) -> None:
        assert self._key(src) in self.objects, f"no such object: {src}"
        self.objects.add(self._key(dst))

    def touch(self, path: str) -> None:
        self.objects.add(self._key(path))

    def rm(self, path: str, recursive: bool = False) -> None:
        key = self._key(path)
        assert recursive or key in self.objects
        self.objects = {
            o for o in self.objects if o != key and not o.startswith(f"{key}/")
        }


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> FakeS3:
    fake = FakeS3()
    for name in ("exists", "find", "ls", "copy", "touch", "rm"):
        monkeypatch.setattr(s3, name, getattr(fake, name))
    return fake


@pytest.fixture
def local_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let etl.load write to local paths, which reject S3 storage_options."""
    monkeypatch.setattr(s3, "s3_opts", lambda: None)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def review(**overrides: Any) -> dict[str, Any]:
    """One raw review, as it appears in the Hugging Face jsonl."""
    row: dict[str, Any] = {
        "user_id": "u1",
        "parent_asin": "B000000001",
        "asin": "B000000001",
        "rating": 5,
        "title": "Nice",
        "text": "Fits well",
        "timestamp": 1_600_000_000_000,  # 2020-09-13 UTC
        "helpful_vote": 0,
        "verified_purchase": True,
        "images": [],
    }
    return row | overrides
