"""Reading the processed datasets and item images, for notebooks and models.

    from recsys.data.load import read_reviews, read_meta, image_urls, fetch_images

    reviews = read_reviews("Amazon_Fashion", columns=["user_id", "parent_asin",
                           "timestamp"], years=range(2018, 2024))
    meta = read_meta("Amazon_Fashion", columns=["parent_asin", "images"])
    urls = image_urls(meta)  # one row per (item, image)
    images = fetch_images(urls.loc[urls["variant"] == "MAIN", "url"])

Reviews are read from S3 every time (pick columns and years to keep it small).
Item metadata is downloaded once into the local cache and read from disk after
that; images are cached there too. The cache lives in `data/cache/` (git-ignored)
unless `CACHE_DIR` is set in `.env` or the environment.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from recsys.data import s3
from recsys.data.paths import dataset_uri
from recsys.settings import get_settings

# Amazon's "no image available" picture, which some items list as their image
PLACEHOLDER = "01RmK+J4pJL"

# columns that identify one review event; exact repeats of it are dropped
DEDUP_KEY = ["user_id", "parent_asin", "asin", "timestamp", "rating"]
# columns the cleaning rules look at
_CLEAN_COLS = ["rating", "helpful_vote"]

Fetcher = Callable[[str], bytes]


# --- reviews -----------------------------------------------------------------


def _reviews_dataset(category: str) -> ds.Dataset:
    path = dataset_uri("reviews", category).removeprefix("s3://")
    return ds.dataset(path, filesystem=s3.filesystem(), partitioning="hive")


def read_reviews(
    category: str,
    columns: Sequence[str] | None = None,
    years: Iterable[int] | None = None,
    clean: bool = True,
) -> pd.DataFrame:
    """Reviews of one category, e.g. category='Amazon_Fashion'.

    `columns=None` reads everything, including review texts, which is slow for
    big categories. `years` keeps only those partitions and skips the rest of
    the files entirely.

    `clean=True` drops rows with a rating outside 1-5 or a negative
    `helpful_vote`, and exact repeats of the same review (same `DEDUP_KEY`).
    The columns these rules need are read even if `columns` leaves them out,
    and then dropped again.
    """
    dataset = _reviews_dataset(category)
    wanted = list(columns) if columns is not None else dataset.schema.names
    extra = [*_CLEAN_COLS, *DEDUP_KEY] if clean else []
    to_read = list(dict.fromkeys([*wanted, *extra]))
    flt = ds.field("year").isin(list(years)) if years is not None else None
    df = dataset.to_table(columns=to_read, filter=flt).to_pandas()
    if clean:
        ok = df["rating"].between(1, 5) & (df["helpful_vote"] >= 0)
        df = df[ok].drop_duplicates(subset=DEDUP_KEY)
    return df[wanted].reset_index(drop=True)


# --- item metadata -------------------------------------------------------------


def cache_dir() -> Path:
    return get_settings().cache_dir


def _meta_cache(category: str) -> Path:
    return cache_dir() / "meta" / category.lower()


def download_meta(category: str, refresh: bool = False) -> Path:
    """Copy the metadata dataset into the local cache, once; returns its folder.

    The local copy counts as complete only when it has its own `_SUCCESS`, so an
    interrupted download is redone on the next call. `refresh=True` downloads
    again, e.g. after the ETL has rebuilt the dataset.
    """
    local = _meta_cache(category)
    if (local / s3.SUCCESS).exists() and not refresh:
        return local
    remote = dataset_uri("meta", category)
    if not s3.exists(f"{remote}/{s3.SUCCESS}"):
        raise FileNotFoundError(f"{remote} is not complete (no {s3.SUCCESS})")
    tmp = local.with_name(f"{local.name}.part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    s3.get(f"{remote}/", f"{tmp}/")
    shutil.rmtree(local, ignore_errors=True)
    tmp.rename(local)
    (local / s3.SUCCESS).touch()
    return local


def read_meta(
    category: str, columns: Sequence[str] | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Item metadata of one category, one row per `parent_asin`.

    The first call downloads the dataset into the cache; later calls read the
    local copy. JSON columns (`images`, `details`, ...) stay as strings.
    """
    local = download_meta(category, refresh=refresh)
    # files starting with "_" (the _SUCCESS marker) are skipped by pyarrow
    dataset = ds.dataset(local, format="parquet")
    cols = list(columns) if columns is not None else None
    return dataset.to_table(columns=cols).to_pandas()


# --- images --------------------------------------------------------------------


def _parse_images(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value:
        return []
    parsed = json.loads(value)
    return parsed if isinstance(parsed, list) else []


def image_urls(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, image): `parent_asin`, `variant` (MAIN, PT01, ...), `url`.

    Takes `hi_res` when the item has it, `large` otherwise. Images without a URL
    and Amazon's placeholder picture are left out, so items without real images
    don't appear at all. Needs the `parent_asin` and `images` columns.
    """
    rows = []
    for asin, value in zip(meta["parent_asin"], meta["images"], strict=True):
        for image in _parse_images(value):
            url = image.get("hi_res") or image.get("large")
            if not url or PLACEHOLDER in url:
                continue
            rows.append((asin, image.get("variant"), url))
    return pd.DataFrame(rows, columns=["parent_asin", "variant", "url"])


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "recsys-eda/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return bytes(response.read())


def _image_path(url: str) -> Path:
    name = hashlib.sha1(url.encode()).hexdigest()
    suffix = Path(url.split("?")[0]).suffix.lower() or ".img"
    return cache_dir() / "images" / name[:2] / f"{name}{suffix}"


def fetch_image(url: str, fetch: Fetcher = _http_get, retries: int = 3) -> bytes:
    """Image bytes, from the cache or downloaded (and then cached).

    Retries timeouts and server errors with a growing pause; a 4xx (e.g. a
    removed image) fails right away. Decode with PIL: Image.open(BytesIO(data)).
    """
    path = _image_path(url)
    if path.exists():
        return path.read_bytes()
    for attempt in range(retries):
        try:
            data = fetch(url)
            break
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.part")
    tmp.write_bytes(data)
    tmp.replace(path)  # atomic, so a half-written file is never read as cached
    return data


def fetch_images(
    urls: Iterable[str], workers: int = 16, fetch: Fetcher = _http_get
) -> dict[str, bytes | None]:
    """Download many images in parallel; a failed image maps to None."""

    def one(url: str) -> bytes | None:
        try:
            return fetch_image(url, fetch=fetch)
        except OSError:  # covers URLError, HTTPError and timeouts
            return None

    unique = list(dict.fromkeys(urls))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(unique, pool.map(one, unique), strict=True))
