"""Where datasets come from and where they live. The single source of truth for
paths, used by the ETL and by notebooks:

    pd.read_parquet(dataset_uri("reviews", "Amazon_Fashion"), storage_options=...)
"""

from typing import Literal

from recsys.settings import get_settings

Kind = Literal["reviews", "meta"]
KINDS: tuple[Kind, ...] = ("reviews", "meta")

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
# Pin to a commit hash for reproducible results; "main" follows the latest version.
HF_REVISION = "main"


def hf_source(kind: Kind, category: str) -> str:
    """hf:// path of the raw jsonl, e.g. category='Amazon_Fashion'."""
    root = f"hf://datasets/{HF_REPO}@{HF_REVISION}/raw"
    if kind == "reviews":
        return f"{root}/review_categories/{category}.jsonl"
    return f"{root}/meta_categories/meta_{category}.jsonl"


def dataset_uri(kind: Kind, category: str) -> str:
    """s3:// folder of the processed dataset, e.g. .../processed/reviews/amazon_fashion"""
    bucket = get_settings().s3_bucket_name
    return f"s3://{bucket}/data/processed/{kind}/{category.lower()}"
