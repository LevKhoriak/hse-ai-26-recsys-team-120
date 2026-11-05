import json
from typing import Any

import pandas as pd
import pyarrow as pa
from conftest import review

from recsys.data.datasets import (
    META_SCHEMA,
    REVIEWS_SCHEMA,
    SPECS,
    transform_meta,
    transform_reviews,
)


def _meta_chunk(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_reviews_replace_images_with_a_count() -> None:
    chunk = pd.DataFrame(
        [
            review(images=[{"url": "a"}, {"url": "b"}]),
            review(images=[]),
            review(images=None),
        ]
    )

    out = transform_reviews(chunk)

    assert "images" not in out.columns
    assert out["n_images"].tolist() == [2, 0, 0]


def test_reviews_get_compact_stable_types() -> None:
    out = transform_reviews(pd.DataFrame([review()]))

    assert out["rating"].dtype == "int8"
    assert out["helpful_vote"].dtype == "int32"
    assert out["n_images"].dtype == "int16"
    assert out["year"].dtype == "int16"


def test_reviews_fit_pinned_schema_even_when_text_is_all_null() -> None:
    out = transform_reviews(pd.DataFrame([review(title=None, text=None)]))

    table = pa.Table.from_pandas(out, schema=REVIEWS_SCHEMA, preserve_index=False)

    assert table.schema.equals(REVIEWS_SCHEMA)
    assert table.column("text").to_pylist() == [None]


def test_reviews_timestamp_is_parsed_from_milliseconds() -> None:
    out = transform_reviews(pd.DataFrame([review(timestamp=1_600_000_000_123)]))

    assert out["timestamp"].iloc[0] == pd.Timestamp("2020-09-13 12:26:40.123")
    assert out["year"].iloc[0] == 2020


def test_meta_price_string_none_becomes_nan() -> None:
    out = transform_meta(
        _meta_chunk(
            {"parent_asin": "A", "price": "None"}, {"parent_asin": "B", "price": 9.5}
        )
    )

    assert pd.isna(out["price"].iloc[0])
    assert out["price"].iloc[1] == 9.5


def test_meta_lists_are_normalized_to_lists_of_strings() -> None:
    out = transform_meta(
        _meta_chunk(
            {"parent_asin": "A", "features": ["x", None, 1]},
            {"parent_asin": "B", "features": None},
        )
    )

    assert out["features"].tolist() == [["x", "1"], []]


def test_meta_nested_fields_become_json_strings() -> None:
    out = transform_meta(
        _meta_chunk(
            {"parent_asin": "A", "details": {"цвет": "red"}},
            {"parent_asin": "B", "details": '{"already": "json"}'},
            {"parent_asin": "C", "details": None},
        )
    )

    assert json.loads(out["details"].iloc[0]) == {"цвет": "red"}
    assert "цвет" in out["details"].iloc[0]  # readable, not \\u-escaped
    assert out["details"].iloc[1] == '{"already": "json"}'
    assert pd.isna(out["details"].iloc[2])


def test_meta_rating_number_stays_integer_with_missing_values() -> None:
    out = transform_meta(
        _meta_chunk(
            {"parent_asin": "A", "rating_number": 3},
            {"parent_asin": "B", "rating_number": None},
        )
    )

    assert out["rating_number"].dtype == "Int64"
    assert out["rating_number"].iloc[0] == 3
    assert pd.isna(out["rating_number"].iloc[1])


def test_meta_fits_pinned_schema_with_missing_and_extra_columns() -> None:
    out = transform_meta(_meta_chunk({"parent_asin": "A", "author": {"name": "x"}}))

    table = pa.Table.from_pandas(out, schema=META_SCHEMA, preserve_index=False)

    assert table.schema.equals(META_SCHEMA)
    assert table.column("title").to_pylist() == [None]


def test_specs_partition_reviews_only() -> None:
    assert SPECS["reviews"].partition_cols == ["year"]
    assert SPECS["reviews"].schema is REVIEWS_SCHEMA
    assert SPECS["meta"].partition_cols == []
    assert SPECS["meta"].schema is META_SCHEMA
