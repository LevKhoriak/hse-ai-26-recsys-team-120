import sys
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from conftest import BUCKET, FakeS3

from recsys.data import cleanup, s3
from recsys.data.s3 import RUN_ID_FORMAT, abort_multipart_uploads

ROOT = f"{BUCKET}/_staging"


def _run_folder(age: timedelta, name: str = "dataset") -> str:
    started = (datetime.now(UTC) - age).strftime(RUN_ID_FORMAT)
    return f"{ROOT}/{name}-{started}"


# --- stale_staging_runs -------------------------------------------------------


def test_no_staging_folder_means_nothing_stale(fake_s3: FakeS3) -> None:
    assert cleanup.stale_staging_runs(BUCKET, timedelta(hours=24)) == []


def test_only_runs_older_than_the_cutoff_are_stale(fake_s3: FakeS3) -> None:
    old = _run_folder(timedelta(hours=48), "old")
    recent = _run_folder(timedelta(hours=1), "recent")
    fake_s3.add(f"{old}/a.parquet", f"{recent}/a.parquet")

    assert cleanup.stale_staging_runs(BUCKET, timedelta(hours=24)) == [old]


def test_folders_with_unknown_names_are_never_stale(fake_s3: FakeS3) -> None:
    fake_s3.add(f"{ROOT}/someones-notes/a.txt", f"{ROOT}/no_dash/a.txt")

    assert cleanup.stale_staging_runs(BUCKET, timedelta(hours=0)) == []


def test_dataset_names_with_dashes_still_parse(fake_s3: FakeS3) -> None:
    old = _run_folder(timedelta(hours=48), "name-with-dashes")
    fake_s3.add(f"{old}/a.parquet")

    assert cleanup.stale_staging_runs(BUCKET, timedelta(hours=24)) == [old]


# --- main ---------------------------------------------------------------------


@pytest.fixture
def aborts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, timedelta]]:
    calls: list[tuple[str, timedelta]] = []

    def abort(bucket: str, older_than: timedelta) -> int:
        calls.append((bucket, older_than))
        return 1

    monkeypatch.setattr(cleanup, "abort_multipart_uploads", abort)
    monkeypatch.setattr(cleanup, "setup_logging", lambda level: None)
    return calls


def test_main_removes_stale_runs_and_aborts_old_uploads(
    fake_s3: FakeS3,
    aborts: list[tuple[str, timedelta]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _run_folder(timedelta(hours=48), "old")
    recent = _run_folder(timedelta(hours=1), "recent")
    fake_s3.add(f"{old}/a.parquet", f"{recent}/a.parquet")
    monkeypatch.setattr(sys, "argv", ["cleanup", "--older-than-hours", "24"])

    cleanup.main()

    assert fake_s3.objects == {f"{recent}/a.parquet"}
    assert aborts == [(BUCKET, timedelta(hours=24))]


def test_main_dry_run_changes_nothing(
    fake_s3: FakeS3,
    aborts: list[tuple[str, timedelta]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _run_folder(timedelta(hours=48))
    fake_s3.add(f"{old}/a.parquet")
    monkeypatch.setattr(sys, "argv", ["cleanup", "--dry-run"])

    cleanup.main()

    assert fake_s3.objects == {f"{old}/a.parquet"}
    assert aborts == []


# --- abort_multipart_uploads --------------------------------------------------


class FakeClient:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.paginate_kwargs: dict[str, Any] = {}
        self.aborted: list[tuple[str, str]] = []

    def get_paginator(self, name: str) -> "FakeClient":
        assert name == "list_multipart_uploads"
        return self

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.paginate_kwargs = kwargs
        return self.pages

    def abort_multipart_upload(self, Bucket: str, Key: str, UploadId: str) -> None:
        self.aborted.append((Key, UploadId))


def _upload(key: str, age: timedelta) -> dict[str, Any]:
    return {"Key": key, "UploadId": f"id-{key}", "Initiated": datetime.now(UTC) - age}


def test_abort_handles_every_page_and_empty_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(
        [
            {"Uploads": [_upload("a", timedelta(hours=1))]},
            {},  # S3 omits Uploads when there are none
            {"Uploads": [_upload("b", timedelta(hours=1))]},
        ]
    )
    monkeypatch.setattr(s3, "_client", lambda: client)

    assert abort_multipart_uploads(BUCKET, prefix="_staging/run") == 2
    assert client.paginate_kwargs == {"Bucket": BUCKET, "Prefix": "_staging/run"}
    assert client.aborted == [("a", "id-a"), ("b", "id-b")]


def test_abort_spares_uploads_newer_than_the_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(
        [
            {
                "Uploads": [
                    _upload("old", timedelta(hours=48)),
                    _upload("running", timedelta(hours=1)),
                ]
            }
        ]
    )
    monkeypatch.setattr(s3, "_client", lambda: client)

    aborted = abort_multipart_uploads(BUCKET, older_than=timedelta(hours=24))

    assert aborted == 1
    assert client.aborted == [("old", "id-old")]
