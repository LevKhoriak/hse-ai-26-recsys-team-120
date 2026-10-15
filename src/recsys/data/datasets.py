"""Per-kind cleanup: how each raw jsonl becomes a parquet dataset.

Every chunk must produce identical column types, otherwise the part files can't
be read back together. Reviews get there with explicit casts; metadata has
nested, irregular fields, so it gets a pinned schema.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow as pa

from recsys.data.paths import Kind


@dataclass(frozen=True)
class DatasetSpec:
    transform: Callable[[pd.DataFrame], pd.DataFrame]
    partition_cols: list[str]  # empty -> plain part files, no partitioning
    schema: pa.Schema | None = None  # pinned output schema, if any


# --- reviews -----------------------------------------------------------------


def transform_reviews(df: pd.DataFrame) -> pd.DataFrame:
    return (
        # keep only the photo count: the full `images` lists get a different parquet
        # type in chunks where every list is empty, which breaks reading back
        df.assign(n_images=lambda d: d["images"].str.len().fillna(0).astype("int16"))
        .drop(columns=["images"])
        .astype({"rating": "int8", "helpful_vote": "int32"})
        .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"], unit="ms"))
        .assign(year=lambda d: d["timestamp"].dt.year.astype("int16"))
    )


# --- item metadata -------------------------------------------------------------

META_SCHEMA = pa.schema(
    [
        ("parent_asin", pa.string()),
        ("main_category", pa.string()),
        ("title", pa.string()),
        ("store", pa.string()),
        ("price", pa.float64()),  # raw values include the string "None"
        ("average_rating", pa.float64()),
        ("rating_number", pa.int64()),
        ("features", pa.list_(pa.string())),
        ("description", pa.list_(pa.string())),
        ("categories", pa.list_(pa.string())),
        # nested and irregular (keys differ per item), stored as JSON strings;
        # parse with json.loads where needed, e.g. in the image pipeline
        ("images", pa.string()),
        ("videos", pa.string()),
        ("details", pa.string()),
        ("bought_together", pa.string()),
    ]
)
_STR_COLS = ["parent_asin", "main_category", "title", "store"]
_LIST_COLS = ["features", "description", "categories"]
_JSON_COLS = ["images", "videos", "details", "bought_together"]


def _to_json(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):  # some fields already arrive JSON-encoded
        return value
    return json.dumps(value, ensure_ascii=False)


def _to_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return []


def transform_meta(df: pd.DataFrame) -> pd.DataFrame:
    # columns missing from a chunk become null; extra ones (e.g. `author`,
    # `subtitle` in some categories) are dropped by the schema on write
    df = df.reindex(columns=list(dict.fromkeys([*META_SCHEMA.names, *df.columns])))
    for col in _STR_COLS:  # also covers columns that are entirely missing
        df[col] = df[col].astype("string")
    for col in _LIST_COLS:
        df[col] = df[col].map(_to_str_list)
    for col in _JSON_COLS:
        df[col] = df[col].map(_to_json)
    return df.assign(
        price=pd.to_numeric(df["price"], errors="coerce"),
        average_rating=pd.to_numeric(df["average_rating"], errors="coerce"),
        rating_number=pd.to_numeric(df["rating_number"], errors="coerce").astype(
            "Int64"
        ),
    )


SPECS: dict[Kind, DatasetSpec] = {
    "reviews": DatasetSpec(transform=transform_reviews, partition_cols=["year"]),
    "meta": DatasetSpec(transform=transform_meta, partition_cols=[], schema=META_SCHEMA),
}
