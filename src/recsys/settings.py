"""Project settings.

Shared, non-secret values (endpoint, region, bucket) are defaults right here,
committed for the whole team. `.env` only needs personal secrets, though any
setting can still be overridden there or by an environment variable.

AWS access keys are deliberately not here: boto3 and s3fs read
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from the environment themselves.
"""

from functools import cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # `.env` is resolved relative to the working directory (the repo root for
    # scripts); real environment variables always take precedence over it
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_endpoint_url: str = "https://s3.ru-7.storage.selcloud.ru"
    aws_region: str = "ru-7"
    s3_bucket_name: str = "hse-recsys-project"


@cache
def get_settings() -> Settings:
    """Read settings on first use, not at import time, so modules import cleanly."""
    return Settings()
