"""The real extract -> transform -> load path, using local files instead of S3."""

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest
from conftest import review, write_jsonl

from recsys.data import etl
from recsys.data.datasets import META_SCHEMA, SPECS
from recsys.data.paths import Kind


def _run(source: Path, kind: Kind, staging: Path | None, chunksize: int = 2) -> int:
    return etl.run(
        source=str(source),
        size=source.stat().st_size,
        spec=SPECS[kind],
        staging=None if staging is None else str(staging),
        chunksize=chunksize,
    )


def test_extract_yields_chunks_and_growing_byte_positions(tmp_path: Path) -> None:
    source = write_jsonl(
        tmp_path / "r.jsonl", [review(user_id=f"u{i}") for i in range(5)]
    )

    chunks = list(etl.extract(str(source), chunksize=2))

    assert [len(chunk) for chunk, _ in chunks] == [2, 2, 1]
    positions = [pos for _, pos in chunks]
    assert positions == sorted(positions)
    assert positions[-1] == source.stat().st_size


def test_extract_keeps_digit_only_ids_as_strings(tmp_path: Path) -> None:
    source = write_jsonl(tmp_path / "r.jsonl", [review(parent_asin="0123456789")])

    chunk, _ = next(etl.extract(str(source), chunksize=10))

    assert chunk["parent_asin"].iloc[0] == "0123456789"


def test_source_size_of_a_local_file(tmp_path: Path) -> None:
    source = write_jsonl(tmp_path / "r.jsonl", [review()])

    assert etl.source_size(str(source)) == source.stat().st_size


@pytest.mark.usefixtures("local_storage")
def test_reviews_round_trip_across_chunks_and_years(tmp_path: Path) -> None:
    year_ms = 366 * 24 * 3600 * 1000
    rows = [
        review(
            user_id=f"u{i}",
            parent_asin="0123456789",
            timestamp=1_600_000_000_000 + (i // 3) * year_ms,
            images=[{"url": "a"}] if i == 0 else [],
        )
        for i in range(6)
    ]
    source = write_jsonl(tmp_path / "r.jsonl", rows)
    staging = tmp_path / "staging"

    total = _run(source, "reviews", staging)

    back = pd.read_parquet(staging)
    assert total == 6
    assert sorted(back["user_id"]) == [f"u{i}" for i in range(6)]
    assert set(back["parent_asin"]) == {"0123456789"}
    assert back["n_images"].sum() == 1
    assert {p.name for p in staging.iterdir()} == {"year=2020", "year=2021"}


@pytest.mark.usefixtures("local_storage")
def test_chunks_in_the_same_year_do_not_overwrite_each_other(tmp_path: Path) -> None:
    source = write_jsonl(
        tmp_path / "r.jsonl", [review(user_id=f"u{i}") for i in range(4)]
    )
    staging = tmp_path / "staging"

    _run(source, "reviews", staging, chunksize=1)

    assert len(list((staging / "year=2020").glob("*.parquet"))) == 4
    assert len(pd.read_parquet(staging)) == 4


@pytest.mark.usefixtures("local_storage")
def test_reviews_chunk_with_all_null_column_stays_readable(tmp_path: Path) -> None:
    # The reader takes its schema from the first file. Within one year, file order
    # depends on random names, so the null chunk goes in the earlier year: that
    # reliably failed before reviews had a pinned schema.
    year_ms = 366 * 24 * 3600 * 1000
    rows = [
        review(
            user_id=f"u{i}",
            text=None if i < 2 else "ok",
            timestamp=1_600_000_000_000 + (i // 2) * year_ms,
        )
        for i in range(4)
    ]
    source = write_jsonl(tmp_path / "r.jsonl", rows)
    staging = tmp_path / "staging"

    _run(source, "reviews", staging)

    assert pd.read_parquet(staging)["text"].notna().sum() == 2


@pytest.mark.usefixtures("local_storage")
def test_meta_writes_numbered_parts_with_the_pinned_schema(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {"parent_asin": "0000000001", "price": "None", "images": [{"url": "é"}]},
        {"parent_asin": "B2", "price": 9.5, "author": {"name": "dropped"}},
        {"parent_asin": "B3", "features": ["f"]},
    ]
    source = write_jsonl(tmp_path / "m.jsonl", rows)
    staging = tmp_path / "staging"
    staging.mkdir()  # s3 has no folders; a local parquet write needs one

    total = _run(source, "meta", staging)

    parts = sorted(p.name for p in staging.iterdir())
    assert total == 3
    assert parts == ["part-00000.parquet", "part-00001.parquet"]
    for part in staging.iterdir():
        assert pq.read_schema(part).remove_metadata().equals(META_SCHEMA)
    back = pd.read_parquet(staging)
    assert sorted(back["parent_asin"]) == ["0000000001", "B2", "B3"]
    assert "author" not in back.columns


def test_run_without_staging_counts_rows_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry run must not write")

    monkeypatch.setattr(etl, "load", fail_load)
    source = write_jsonl(tmp_path / "r.jsonl", [review() for _ in range(3)])

    assert _run(source, "reviews", staging=None) == 3


def test_run_on_an_empty_source_counts_zero_rows(tmp_path: Path) -> None:
    source = tmp_path / "r.jsonl"
    source.touch()

    assert _run(source, "reviews", staging=None) == 0


def test_run_propagates_malformed_input(tmp_path: Path) -> None:
    source = tmp_path / "r.jsonl"
    source.write_text('{"user_id": \n', encoding="utf-8")

    with pytest.raises(ValueError):
        _run(source, "reviews", staging=None)
