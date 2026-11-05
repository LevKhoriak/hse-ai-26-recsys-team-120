from conftest import BUCKET

from recsys.data.paths import dataset_uri, hf_source
from recsys.data.s3 import s3_opts, split


def test_hf_source_reviews() -> None:
    assert hf_source("reviews", "Amazon_Fashion").endswith(
        "/raw/review_categories/Amazon_Fashion.jsonl"
    )


def test_hf_source_meta() -> None:
    assert hf_source("meta", "Amazon_Fashion").endswith(
        "/raw/meta_categories/meta_Amazon_Fashion.jsonl"
    )


def test_dataset_uri_lowercases_category_and_uses_bucket_from_settings() -> None:
    assert (
        dataset_uri("reviews", "Grocery_and_Gourmet_Food")
        == f"s3://{BUCKET}/data/processed/reviews/grocery_and_gourmet_food"
    )


def test_split_handles_prefix_and_trailing_slash() -> None:
    assert split("s3://bucket/a/b/") == ("bucket", "a/b")
    assert split("bucket/a/b") == ("bucket", "a/b")
    assert split("s3://bucket") == ("bucket", "")


def test_s3_opts_come_from_settings() -> None:
    assert s3_opts() == {
        "endpoint_url": "https://s3.test",
        "client_kwargs": {"region_name": "test-region"},
    }
