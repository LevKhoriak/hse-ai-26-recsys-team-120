"""Remove leftovers of crashed or killed ETL runs.

etl.py cleans up after itself on errors and Ctrl+C. This is the backstop for
runs that died without a chance to (laptop closed, kill -9, lost connection).
"""

import argparse
from datetime import UTC, datetime, timedelta

from recsys.data import s3
from recsys.data.s3 import (
    RUN_ID_FORMAT,
    STAGING,
    abort_multipart_uploads,
)
from recsys.logging import get_logger, setup_logging
from recsys.settings import get_settings

log = get_logger("cleanup")


def stale_staging_runs(bucket: str, older_than: timedelta) -> list[str]:
    root = f"{bucket}/{STAGING}"
    if not s3.exists(root):
        return []
    cutoff = datetime.now(UTC) - older_than
    stale = []
    for path in s3.ls(root):
        run_id = path.rstrip("/").rsplit("-", 1)[-1]  # <name>-<run_id>
        try:
            started = datetime.strptime(run_id, RUN_ID_FORMAT).replace(tzinfo=UTC)
        except ValueError:
            log.warning("skipping %s: not an etl staging folder", path)
            continue
        if started < cutoff:
            stale.append(path)
    return stale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", help="defaults to S3_BUCKET_NAME from settings")
    parser.add_argument(
        "--older-than-hours",
        type=float,
        default=24,
        help="only touch leftovers at least this old, so running jobs are safe",
    )
    parser.add_argument("--dry-run", action="store_true", help="only list")
    args = parser.parse_args()
    bucket: str = args.bucket or get_settings().s3_bucket_name
    setup_logging("INFO")
    age = timedelta(hours=args.older_than_hours)

    stale = stale_staging_runs(bucket, age)
    for path in stale:
        log.info("%s stale staging run %s", "would remove" if args.dry_run else "removing", path)
        if not args.dry_run:
            s3.rm(path, recursive=True)

    if args.dry_run:
        log.info("dry run: multipart uploads not aborted")
    else:
        n = abort_multipart_uploads(bucket, older_than=age)
        log.info("aborted %d unfinished multipart uploads", n)


if __name__ == "__main__":
    main()
