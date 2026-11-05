"""HF jsonl -> cleaned parquet on S3.

    uv run python src/recsys/data/etl.py --kind reviews --category Amazon_Fashion
    uv run python src/recsys/data/etl.py --kind meta --category Amazon_Fashion

Writes go to a staging folder first and replace the dataset only on success.
"""

import argparse
import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import BinaryIO, cast

import fsspec
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from recsys.data import s3
from recsys.data.datasets import SPECS, DatasetSpec
from recsys.data.paths import KINDS, Kind, dataset_uri, hf_source
from recsys.data.s3 import (
    BACKUP,
    STAGING,
    SUCCESS,
    abort_multipart_uploads,
    new_run_id,
    split,
)
from recsys.logging import get_logger, setup_logging

log = get_logger("etl")


# fsspec ships no usable types either (its open() looks like it returns a list),
# so the two calls we make to it are wrapped here with explicit types.


@contextmanager
def open_binary(path: str) -> Iterator[BinaryIO]:
    """Open any fsspec path (hf://, s3://, local) for binary reading."""
    with fsspec.open(path, "rb") as f:
        yield cast(BinaryIO, f)


def source_size(path: str) -> int:
    fs, fs_path = fsspec.core.url_to_fs(path)
    return cast(int, fs.size(fs_path))


# --- extract / load ---------------------------------------------------------------


def extract(hf_path: str, chunksize: int) -> Iterator[tuple[pd.DataFrame, int]]:
    """Yield (chunk, bytes read so far) so progress can be measured in bytes."""
    with (
        open_binary(hf_path) as raw,
        io.TextIOWrapper(raw, encoding="utf-8") as text,
        pd.read_json(
            text,
            lines=True,
            chunksize=chunksize,
            # types are set explicitly in the transforms; without dtype=False pandas
            # may turn all-digit ids (e.g. ISBN-like ASINs) into numbers
            dtype=False,
            convert_dates=False,
        ) as reader,
    ):
        for chunk in reader:
            yield chunk, raw.tell()


def load(df: pd.DataFrame, spec: DatasetSpec, staging: str, part: int) -> None:
    opts = s3.s3_opts()
    if spec.partition_cols:
        # adds uniquely named files under e.g. year=YYYY/
        df.to_parquet(
            staging,
            partition_cols=spec.partition_cols,
            index=False,
            schema=spec.schema,
            storage_options=opts,
        )
    else:
        df.to_parquet(
            f"{staging}/part-{part:05d}.parquet",
            index=False,
            schema=spec.schema,
            storage_options=opts,
        )


# --- staging and publishing ----------------------------------------------------


def _run_folder(prefix: str, dest: str) -> str:
    bucket, key = split(dest)
    name = key.replace("/", "_")  # e.g. data_processed_reviews_amazon_fashion
    return f"s3://{bucket}/{prefix}/{name}-{new_run_id()}"


def staging_path(dest: str) -> str:
    """Each run writes to its own folder under s3://<bucket>/_staging/ first."""
    return _run_folder(STAGING, dest)


def _copy_tree(src: str, dest: str) -> None:
    """Copy a folder, with the _SUCCESS marker last so it's only there when complete."""
    src_root = src.removeprefix("s3://").rstrip("/")
    dest_root = dest.removeprefix("s3://").rstrip("/")
    files = sorted(s3.find(src_root), key=lambda path: path.endswith(f"/{SUCCESS}"))
    for path in files:
        s3.copy(path, dest_root + path[len(src_root) :])


def _back_up(dest: str) -> str:
    """Copy dest aside. _backup/ is out of cleanup.py's reach, on purpose: after a
    hard crash mid-publish it may hold the only complete version of the dataset."""
    backup = _run_folder(BACKUP, dest)
    try:
        _copy_tree(dest, backup)
    except BaseException:
        if s3.exists(backup):
            s3.rm(backup, recursive=True)
        raise
    return backup


def _roll_back(dest: str, backup: str | None) -> None:
    """Undo a failed publish: drop partial output and restore the old version."""
    try:
        if s3.exists(dest):
            s3.rm(dest, recursive=True)
        if backup is not None:
            _copy_tree(backup, dest)
            s3.rm(backup, recursive=True)
    except Exception:
        log.exception("rollback failed: %s is incomplete, see %s", dest, backup)


def publish(staging: str, dest: str) -> None:
    """Replace dest with the finished staging folder, then mark it complete.

    S3 can't swap folders atomically, so the old version is backed up first and
    restored if anything fails. Readers can tell a replacement is in progress
    because _SUCCESS is missing. On failure staging is kept for cleanup.py.
    """
    backup = None
    if s3.exists(dest):
        log.info("replacing existing %s", dest)
        backup = _back_up(dest)
    try:
        if backup is not None:
            s3.rm(dest, recursive=True)
        _copy_tree(staging, dest)
        s3.touch(f"{dest.removeprefix('s3://').rstrip('/')}/{SUCCESS}")
    except BaseException:  # includes Ctrl+C
        log.error("publish failed, rolling back %s", dest)
        _roll_back(dest, backup)
        raise
    s3.rm(staging, recursive=True)
    if backup is not None:
        s3.rm(backup, recursive=True)


def discard(staging: str) -> None:
    """Remove a failed run's partial output and its unfinished uploads."""
    bucket, key = split(staging)
    n = abort_multipart_uploads(bucket, prefix=key)
    if s3.exists(staging):
        s3.rm(staging, recursive=True)
    log.info("cleaned up %s (%d unfinished uploads aborted)", staging, n)


# --- cli ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument(
        "--category",
        required=True,
        help="as in the HF file names, e.g. Amazon_Fashion, Grocery_and_Gourmet_Food",
    )
    parser.add_argument(
        "--chunksize", type=int, default=200_000, help="rows per chunk"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the dataset if it exists (only after the new run succeeds)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="extract + transform, skip upload"
    )
    return parser.parse_args()


def main() -> None:
    setup_logging("INFO")
    args = parse_args()
    kind: Kind = args.kind
    source = hf_source(kind, args.category)
    dest = dataset_uri(kind, args.category)

    try:
        size = source_size(source)
    except FileNotFoundError:
        raise SystemExit(
            f"no {kind} file for category {args.category!r} ({source}); "
            "check the spelling against the HF repo's file names"
        ) from None

    if not args.dry_run and not args.overwrite and s3.exists(dest):
        # appending would duplicate rows, so refuse before doing any work
        raise SystemExit(f"{dest} already exists, pass --overwrite to replace it")

    staging = staging_path(dest)
    log.info("%s %s: %s -> %s", kind, args.category, source, dest)
    try:
        total = run(
            source=source,
            size=size,
            spec=SPECS[kind],
            staging=None if args.dry_run else staging,
            chunksize=args.chunksize,
        )
        if total == 0:
            # most likely a broken source; never replace real data with nothing
            raise SystemExit(f"{source} has no rows, nothing published to {dest}")
    except BaseException:  # includes Ctrl+C
        if not args.dry_run:
            log.error("run failed, removing partial output")
            discard(staging)
        raise

    if args.dry_run:
        log.info("done: %d rows counted (dry run)", total)
    else:
        publish(staging, dest)
        log.info("done: %d rows written to %s", total, dest)


def run(
    source: str, size: int, spec: DatasetSpec, staging: str | None, chunksize: int
) -> int:
    """Stream source through the transform; write to staging unless it's None."""
    total = 0
    # route log lines through tqdm so they don't break the progress bar
    with (
        logging_redirect_tqdm(loggers=[logging.getLogger("recsys")]),
        tqdm(
            total=size, unit="B", unit_scale=True, unit_divisor=1024, desc="etl"
        ) as bar,
    ):
        for i, (chunk, pos) in enumerate(extract(source, chunksize)):
            df = spec.transform(chunk)
            if staging is not None:
                load(df, spec, staging, part=i)
            total += len(df)
            bar.update(pos - bar.n)
            bar.set_postfix(rows=f"{total:,}")
            log.debug("chunk %d: %d rows", i, len(df))
    return total


if __name__ == "__main__":
    main()
