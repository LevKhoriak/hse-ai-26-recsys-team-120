"""recsys.data.load against local parquet files and a fake HTTP client."""

import json
import shutil
import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import urllib3

from recsys.data import load, s3
from recsys.settings import get_settings


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    get_settings.cache_clear()
    monkeypatch.setattr(
        "recsys.data.load.time.sleep", lambda _: None
    )  # no retry pauses
    return tmp_path / "cache"


# --- reviews -----------------------------------------------------------------


def _review(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "user_id": "u1",
        "parent_asin": "p1",
        "asin": "a1",
        "rating": 5,
        "helpful_vote": 0,
        "timestamp": pd.Timestamp("2020-01-01"),
        "year": 2020,
    }
    return row | overrides


@pytest.fixture
def reviews_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _review(),
        _review(),  # exact repeat
        _review(user_id="u2", rating=0),  # rating out of range
        _review(user_id="u3", helpful_vote=-1),
        _review(user_id="u4", timestamp=pd.Timestamp("2021-05-01"), year=2021),
        _review(user_id="u1", asin="a2"),  # same item, other variant: kept
    ]
    root = tmp_path / "reviews"
    pq.write_to_dataset(pa.Table.from_pylist(rows), root, partition_cols=["year"])
    monkeypatch.setattr(
        load,
        "_reviews_dataset",
        lambda category: ds.dataset(root, partitioning="hive"),
    )


@pytest.mark.usefixtures("reviews_on_disk")
def test_read_reviews_cleans_and_keeps_only_requested_columns() -> None:
    df = load.read_reviews("X", columns=["user_id", "parent_asin"])
    assert list(df.columns) == ["user_id", "parent_asin"]
    assert sorted(df["user_id"]) == ["u1", "u1", "u4"]


@pytest.mark.usefixtures("reviews_on_disk")
def test_read_reviews_without_cleaning_keeps_everything() -> None:
    assert len(load.read_reviews("X", clean=False)) == 6


@pytest.mark.usefixtures("reviews_on_disk")
def test_read_reviews_filters_years() -> None:
    df = load.read_reviews("X", columns=["user_id", "year"], years=[2021])
    assert df["user_id"].tolist() == ["u4"]


# --- metadata ----------------------------------------------------------------


@pytest.fixture
def downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A meta 'bucket' folder on disk, which s3.get copies from; returns the
    log of downloads."""
    remote = tmp_path / "remote"
    remote.mkdir()
    table = pa.Table.from_pylist([{"parent_asin": "p1", "title": "Hat"}])
    pq.write_table(table, remote / "part-00000.parquet")
    (remote / s3.SUCCESS).touch()
    calls: list[str] = []

    def get(src: str, dst: str) -> None:
        calls.append(src)
        shutil.copytree(remote, dst, dirs_exist_ok=True)

    monkeypatch.setattr(s3, "exists", lambda path: (remote / s3.SUCCESS).exists())
    monkeypatch.setattr(s3, "get", get)
    return calls


def test_read_meta_downloads_once_then_reads_cache(
    downloads: list[str], cache: Path
) -> None:
    assert load.read_meta("Amazon_Fashion", columns=["title"])["title"].tolist() == [
        "Hat"
    ]
    load.read_meta("Amazon_Fashion")
    assert len(downloads) == 1
    assert (cache / "meta" / "amazon_fashion" / s3.SUCCESS).exists()


def test_read_meta_refresh_downloads_again(downloads: list[str]) -> None:
    load.read_meta("X")
    load.read_meta("X", refresh=True)
    assert len(downloads) == 2


def test_unfinished_download_is_redone(downloads: list[str], cache: Path) -> None:
    (cache / "meta" / "x").mkdir(parents=True)  # left behind, no _SUCCESS
    load.read_meta("X")
    assert len(downloads) == 1


@pytest.mark.usefixtures("downloads")
def test_incomplete_remote_dataset_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "remote" / s3.SUCCESS).unlink()
    with pytest.raises(FileNotFoundError):
        load.read_meta("X")


# --- images ------------------------------------------------------------------


def test_image_urls_prefers_hi_res_and_drops_placeholders() -> None:
    images = [
        {"variant": "MAIN", "hi_res": "https://x/hi.jpg", "large": "https://x/l.jpg"},
        {"variant": "PT01", "hi_res": None, "large": "https://x/l2.jpg"},
        {
            "variant": "PT02",
            "hi_res": None,
            "large": f"https://x/{load.PLACEHOLDER}.gif",
        },
        {"variant": "PT03", "hi_res": None, "large": None},
    ]
    meta = pd.DataFrame(
        {"parent_asin": ["p1", "p2", "p3"], "images": [json.dumps(images), "[]", None]}
    )
    urls = load.image_urls(meta)
    assert urls.to_dict("records") == [
        {"parent_asin": "p1", "variant": "MAIN", "url": "https://x/hi.jpg"},
        {"parent_asin": "p1", "variant": "PT01", "url": "https://x/l2.jpg"},
    ]
    large = load.image_urls(meta, size="large")
    assert large["url"].tolist() == ["https://x/l.jpg", "https://x/l2.jpg"]


class FakeHttp:
    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.failures = failures or []
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if self.failures:
            raise self.failures.pop(0)
        return f"bytes of {url}".encode()


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", Message(), None)


def test_fetch_image_caches() -> None:
    http = FakeHttp()
    first = load.fetch_image("https://x/a.jpg", fetch=http)
    second = load.fetch_image("https://x/a.jpg", fetch=http)
    assert first == second == b"bytes of https://x/a.jpg"
    assert len(http.calls) == 1


def test_fetch_image_retries_server_errors() -> None:
    http = FakeHttp([_http_error(503), TimeoutError()])
    assert load.fetch_image("https://x/a.jpg", fetch=http)
    assert len(http.calls) == 3


def test_fetch_image_does_not_retry_missing_images() -> None:
    http = FakeHttp([_http_error(404)])
    with pytest.raises(urllib.error.HTTPError):
        load.fetch_image("https://x/a.jpg", fetch=http)
    assert len(http.calls) == 1


def test_fetch_images_maps_failures_to_none() -> None:
    http = FakeHttp([_http_error(404)])
    result = load.fetch_images(["https://x/a.jpg"], workers=1, fetch=http)
    assert result == {"https://x/a.jpg": None}
    ok = load.fetch_images(["https://x/b.jpg", "https://x/b.jpg"], fetch=http)
    assert ok == {"https://x/b.jpg": b"bytes of https://x/b.jpg"}


class FakePool:
    def __init__(self, status: int = 200, error: Exception | None = None) -> None:
        self.status, self.error = status, error

    def request(self, method: str, url: str) -> SimpleNamespace:
        if self.error:
            raise self.error
        return SimpleNamespace(status=self.status, reason="x", data=b"img")


def test_http_get_maps_errors_to_urllib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(load, "_pool", lambda: FakePool())
    assert load._http_get("https://x/a.jpg") == b"img"

    monkeypatch.setattr(load, "_pool", lambda: FakePool(status=404))
    with pytest.raises(urllib.error.HTTPError) as error:
        load._http_get("https://x/a.jpg")
    assert error.value.code == 404

    timeout = urllib3.exceptions.ReadTimeoutError(None, "https://x", "slow")  # type: ignore[arg-type]
    monkeypatch.setattr(load, "_pool", lambda: FakePool(error=timeout))
    with pytest.raises(urllib.error.URLError):  # retried by fetch_image
        load._http_get("https://x/a.jpg")
