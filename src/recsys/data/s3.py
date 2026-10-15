"""Shared S3 helpers: one place for the endpoint, path parsing and cleanup."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import cache
from typing import TYPE_CHECKING, cast

import boto3
import s3fs

from recsys.settings import get_settings

if TYPE_CHECKING:  # stubs are a dev dependency, not needed at runtime
    from types_boto3_s3 import S3Client

STAGING = "_staging"  # bucket-level prefix for in-progress runs
SUCCESS = "_SUCCESS"  # marker written when a dataset is complete (Hadoop-style)
RUN_ID_FORMAT = "%Y%m%dT%H%M%SZ"


def s3_opts() -> dict[str, object]:
    """Connection options for s3fs and pandas `storage_options`.

    (Polars uses different option names: aws_endpoint_url, aws_region.)
    """
    settings = get_settings()
    return {
        "endpoint_url": settings.aws_endpoint_url,
        "client_kwargs": {"region_name": settings.aws_region},
    }


def _client() -> S3Client:
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.aws_endpoint_url,
        region_name=settings.aws_region,
    )


# --- typed wrappers around s3fs -------------------------------------------
# s3fs/fsspec ship no type information (their methods look like `-> None` to
# type checkers), so every s3fs call goes through these and the casts live here.
# Paths may be given with or without the s3:// prefix.


@cache
def _fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(**s3_opts())


def exists(path: str) -> bool:
    return cast(bool, _fs().exists(path))


def ls(path: str) -> list[str]:
    """Immediate children of a folder, as 'bucket/key' paths."""
    return cast(list[str], _fs().ls(path, detail=False))


def find(path: str) -> list[str]:
    """All files under a folder, recursively, as 'bucket/key' paths."""
    return cast(list[str], _fs().find(path))


def copy(src: str, dst: str) -> None:
    """Server-side copy of a single object, nothing is downloaded."""
    _fs().copy(src, dst)


def touch(path: str) -> None:
    _fs().touch(path)


def rm(path: str, recursive: bool = False) -> None:
    _fs().rm(path, recursive=recursive)


# --- paths and cleanup ------------------------------------------------------


def split(s3_path: str) -> tuple[str, str]:
    """'s3://bucket/a/b' -> ('bucket', 'a/b')"""
    bucket, _, key = s3_path.removeprefix("s3://").partition("/")
    return bucket, key.rstrip("/")


def new_run_id() -> str:
    return datetime.now(UTC).strftime(RUN_ID_FORMAT)


def abort_multipart_uploads(
    bucket: str, prefix: str = "", older_than: timedelta | None = None
) -> int:
    """Abort unfinished multipart uploads so their parts stop taking up storage.

    `older_than` protects uploads that a teammate's job may still be running.
    """
    client = _client()
    cutoff = datetime.now(UTC) - older_than if older_than else None
    aborted = 0
    paginator = client.get_paginator("list_multipart_uploads")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for upload in page.get("Uploads", []):
            if cutoff and upload["Initiated"] > cutoff:
                continue
            client.abort_multipart_upload(
                Bucket=bucket, Key=upload["Key"], UploadId=upload["UploadId"]
            )
            aborted += 1
    return aborted
