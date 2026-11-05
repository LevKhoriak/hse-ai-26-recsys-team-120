"""Staging, publishing and the command line, against an in-memory S3."""

import sys
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest
from conftest import BUCKET, FakeS3

from recsys.data import cleanup, etl, s3

DEST = f"s3://{BUCKET}/data/processed/reviews/amazon_fashion"
STAGING = f"s3://{BUCKET}/_staging/run-1"
DEST_KEY = DEST.removeprefix("s3://")
STAGING_KEY = STAGING.removeprefix("s3://")


# --- staging_path -------------------------------------------------------------


def test_staging_path_is_flat_and_under_the_staging_prefix() -> None:
    path = etl.staging_path(DEST)

    prefix = f"s3://{BUCKET}/_staging/data_processed_reviews_amazon_fashion-"
    assert path.startswith(prefix)
    assert "/" not in path.removeprefix(prefix)


def test_cleanup_recognizes_folders_created_by_staging_path(fake_s3: FakeS3) -> None:
    """etl.py writes the names and cleanup.py parses them: they must agree."""
    fake_s3.add(f"{etl.staging_path(DEST)}/part.parquet")

    stale = cleanup.stale_staging_runs(BUCKET, older_than=timedelta(hours=-1))

    assert len(stale) == 1


# --- publish ------------------------------------------------------------------


def test_publish_moves_staging_to_dest_and_marks_success(fake_s3: FakeS3) -> None:
    fake_s3.add(
        f"{STAGING}/year=2020/a.parquet",
        f"{STAGING}/year=2021/b.parquet",
    )

    etl.publish(STAGING, DEST)

    assert fake_s3.objects == {
        f"{DEST_KEY}/year=2020/a.parquet",
        f"{DEST_KEY}/year=2021/b.parquet",
        f"{DEST_KEY}/_SUCCESS",
    }


def test_publish_replaces_previous_version_completely(fake_s3: FakeS3) -> None:
    fake_s3.add(f"{DEST}/year=2019/old.parquet", f"{DEST}/_SUCCESS")
    fake_s3.add(f"{STAGING}/year=2020/new.parquet")

    etl.publish(STAGING, DEST)

    assert f"{DEST_KEY}/year=2019/old.parquet" not in fake_s3.objects
    assert f"{DEST_KEY}/year=2020/new.parquet" in fake_s3.objects


def _fail_on(fake_s3: FakeS3, monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make s3.copy fail for any source file whose name ends with one of names."""
    real_copy = fake_s3.copy

    def flaky_copy(src: str, dst: str) -> None:
        if src.endswith(names):
            raise OSError("connection lost")
        real_copy(src, dst)

    monkeypatch.setattr(s3, "copy", flaky_copy)


def _under(fake_s3: FakeS3, prefix: str) -> set[str]:
    return {o for o in fake_s3.objects if o.startswith(prefix)}


OLD_VERSION = {f"{DEST_KEY}/year=2019/old.parquet", f"{DEST_KEY}/_SUCCESS"}
BACKUP_PREFIX = f"{BUCKET}/_backup/"


def test_publish_leaves_no_backup_after_success(fake_s3: FakeS3) -> None:
    fake_s3.add(*OLD_VERSION, f"{STAGING}/new.parquet")

    etl.publish(STAGING, DEST)

    assert _under(fake_s3, BACKUP_PREFIX) == set()


def test_publish_keeps_old_version_until_it_is_backed_up(
    fake_s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3.add(*OLD_VERSION, f"{STAGING}/new.parquet")
    _fail_on(fake_s3, monkeypatch, "old.parquet")

    with pytest.raises(OSError):
        etl.publish(STAGING, DEST)

    assert _under(fake_s3, DEST_KEY) == OLD_VERSION
    assert _under(fake_s3, BACKUP_PREFIX) == set()
    assert f"{STAGING_KEY}/new.parquet" in fake_s3.objects


def test_publish_failure_restores_old_version(
    fake_s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3.add(*OLD_VERSION, f"{STAGING}/a.parquet", f"{STAGING}/b.parquet")
    _fail_on(fake_s3, monkeypatch, "b.parquet")

    with pytest.raises(OSError):
        etl.publish(STAGING, DEST)

    assert _under(fake_s3, DEST_KEY) == OLD_VERSION
    assert _under(fake_s3, BACKUP_PREFIX) == set()
    assert f"{STAGING_KEY}/b.parquet" in fake_s3.objects


def test_publish_interrupted_by_ctrl_c_restores_old_version(
    fake_s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3.add(*OLD_VERSION, f"{STAGING}/a.parquet")
    real_copy = fake_s3.copy

    def interrupted(src: str, dst: str) -> None:
        if src.endswith("a.parquet"):
            raise KeyboardInterrupt
        real_copy(src, dst)

    monkeypatch.setattr(s3, "copy", interrupted)

    with pytest.raises(KeyboardInterrupt):
        etl.publish(STAGING, DEST)

    assert _under(fake_s3, DEST_KEY) == OLD_VERSION


def test_publish_failure_without_old_version_leaves_nothing_at_dest(
    fake_s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3.add(f"{STAGING}/a.parquet", f"{STAGING}/b.parquet")
    _fail_on(fake_s3, monkeypatch, "b.parquet")

    with pytest.raises(OSError):
        etl.publish(STAGING, DEST)

    assert _under(fake_s3, DEST_KEY) == set()
    assert f"{STAGING_KEY}/b.parquet" in fake_s3.objects


def test_failed_rollback_keeps_backup_and_reports_the_original_error(
    fake_s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3.add(*OLD_VERSION, f"{STAGING}/a.parquet")
    real_copy = fake_s3.copy

    def flaky_copy(src: str, dst: str) -> None:
        if src.endswith("a.parquet"):
            raise OSError("publish failed")
        if src.startswith(BACKUP_PREFIX):
            raise TimeoutError("restore failed")
        real_copy(src, dst)

    monkeypatch.setattr(s3, "copy", flaky_copy)

    with pytest.raises(OSError, match="publish failed"):
        etl.publish(STAGING, DEST)

    assert {o.rsplit("/", 1)[-1] for o in _under(fake_s3, BACKUP_PREFIX)} == {
        "old.parquet",
        "_SUCCESS",
    }


def test_restore_writes_the_success_marker_last(
    fake_s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3.add(*OLD_VERSION, f"{STAGING}/a.parquet")
    copied: list[str] = []
    real_copy = fake_s3.copy

    def recording_copy(src: str, dst: str) -> None:
        if src.endswith("a.parquet"):
            raise OSError("connection lost")
        copied.append(dst)
        real_copy(src, dst)

    monkeypatch.setattr(s3, "copy", recording_copy)

    with pytest.raises(OSError):
        etl.publish(STAGING, DEST)

    restored = [path for path in copied if path.startswith(DEST_KEY)]
    assert restored[-1] == f"{DEST_KEY}/_SUCCESS"
    assert len(restored) == 2


def test_cleanup_does_not_touch_backups(fake_s3: FakeS3) -> None:
    fake_s3.add(f"{BUCKET}/_backup/old-20200101T000000Z/a.parquet")

    assert cleanup.stale_staging_runs(BUCKET, older_than=timedelta(hours=-1)) == []


# --- discard ------------------------------------------------------------------


def test_discard_removes_partial_output_and_aborts_uploads(
    fake_s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def abort(bucket: str, prefix: str = "") -> int:
        calls.append((bucket, prefix))
        return 2

    monkeypatch.setattr(etl, "abort_multipart_uploads", abort)
    fake_s3.add(f"{STAGING}/a.parquet", f"{DEST}/_SUCCESS")

    etl.discard(STAGING)

    assert calls == [(BUCKET, "_staging/run-1")]
    assert fake_s3.objects == {f"{DEST_KEY}/_SUCCESS"}


def test_discard_tolerates_nothing_having_been_written(
    fake_s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(etl, "abort_multipart_uploads", lambda bucket, prefix: 0)

    etl.discard(STAGING)

    assert fake_s3.objects == set()


# --- main ---------------------------------------------------------------------


class Pipeline:
    """Replaces the expensive steps of etl.main and records what was called."""

    def __init__(self) -> None:
        self.run_kwargs: dict[str, Any] | None = None
        self.published: tuple[str, str] | None = None
        self.discarded: str | None = None
        self.run_error: BaseException | None = None
        self.rows = 3


@pytest.fixture
def pipeline(fake_s3: FakeS3, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    state = Pipeline()

    def run(**kwargs: Any) -> int:
        state.run_kwargs = kwargs
        if state.run_error is not None:
            raise state.run_error
        return state.rows

    def publish(staging: str, dest: str) -> None:
        state.published = (staging, dest)

    def discard(staging: str) -> None:
        state.discarded = staging

    monkeypatch.setattr(etl, "run", run)
    monkeypatch.setattr(etl, "publish", publish)
    monkeypatch.setattr(etl, "discard", discard)
    monkeypatch.setattr(etl, "source_size", lambda path: 100)
    monkeypatch.setattr(etl, "setup_logging", lambda level: None)
    return state


MainRunner = Callable[..., None]


@pytest.fixture
def run_main(monkeypatch: pytest.MonkeyPatch) -> MainRunner:
    def _run(*flags: str) -> None:
        argv = ["etl", "--kind", "reviews", "--category", "Amazon_Fashion", *flags]
        monkeypatch.setattr(sys, "argv", argv)
        etl.main()

    return _run


def test_main_runs_then_publishes_to_the_dataset_folder(
    pipeline: Pipeline, run_main: MainRunner
) -> None:
    run_main()

    assert pipeline.run_kwargs is not None
    staging = pipeline.run_kwargs["staging"]
    assert staging.startswith(f"s3://{BUCKET}/_staging/")
    assert pipeline.published == (staging, DEST)
    assert pipeline.discarded is None


def test_main_refuses_existing_dataset_before_doing_any_work(
    pipeline: Pipeline, fake_s3: FakeS3, run_main: MainRunner
) -> None:
    fake_s3.add(f"{DEST}/_SUCCESS")

    with pytest.raises(SystemExit, match="--overwrite"):
        run_main()

    assert pipeline.run_kwargs is None
    assert pipeline.published is None


def test_main_overwrite_replaces_existing_dataset(
    pipeline: Pipeline, fake_s3: FakeS3, run_main: MainRunner
) -> None:
    fake_s3.add(f"{DEST}/_SUCCESS")

    run_main("--overwrite")

    assert pipeline.published is not None


def test_main_dry_run_never_writes(
    pipeline: Pipeline, fake_s3: FakeS3, run_main: MainRunner
) -> None:
    fake_s3.add(f"{DEST}/_SUCCESS")  # an existing dataset doesn't block a dry run

    run_main("--dry-run")

    assert pipeline.run_kwargs is not None
    assert pipeline.run_kwargs["staging"] is None
    assert pipeline.published is None
    assert pipeline.discarded is None


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt()])
def test_main_discards_partial_output_on_failure_and_keeps_old_data(
    pipeline: Pipeline, run_main: MainRunner, error: BaseException
) -> None:
    pipeline.run_error = error

    with pytest.raises(type(error)):
        run_main("--overwrite")

    assert pipeline.run_kwargs is not None
    assert pipeline.discarded == pipeline.run_kwargs["staging"]
    assert pipeline.published is None


def test_main_dry_run_failure_has_nothing_to_discard(
    pipeline: Pipeline, run_main: MainRunner
) -> None:
    pipeline.run_error = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_main("--dry-run")

    assert pipeline.discarded is None


def test_main_empty_source_discards_staging_and_keeps_existing_dataset(
    pipeline: Pipeline, fake_s3: FakeS3, run_main: MainRunner
) -> None:
    fake_s3.add(f"{DEST}/_SUCCESS")
    pipeline.rows = 0

    with pytest.raises(SystemExit, match="no rows"):
        run_main("--overwrite")

    assert pipeline.run_kwargs is not None
    assert pipeline.published is None
    assert pipeline.discarded == pipeline.run_kwargs["staging"]


def test_main_dry_run_reports_empty_source_without_discarding(
    pipeline: Pipeline, run_main: MainRunner
) -> None:
    pipeline.rows = 0

    with pytest.raises(SystemExit, match="no rows"):
        run_main("--dry-run")

    assert pipeline.discarded is None


def test_main_reports_unknown_category(
    pipeline: Pipeline, run_main: MainRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(path: str) -> int:
        raise FileNotFoundError(path)

    monkeypatch.setattr(etl, "source_size", missing)

    with pytest.raises(SystemExit, match="Amazon_Fashion"):
        run_main()

    assert pipeline.run_kwargs is None
