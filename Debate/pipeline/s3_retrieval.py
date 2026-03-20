"""S3 discovery and download helpers for nuPlan logs."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Iterator

import boto3
from botocore.client import BaseClient


@dataclass(frozen=True)
class S3LogObject:
    """One .db log object discovered in S3."""

    bucket: str
    key: str
    log_id: str


def make_s3_client() -> BaseClient:
    """Purpose: Build an S3 client using default credential chain.
    Parameters: None.
    Returns: BaseClient configured boto3 S3 client.
    Called by: CLI entrypoints in retrieval/extraction scripts.
    Calls: boto3.client().
    """
    return boto3.client("s3")


def list_log_db_objects(
    s3_client: BaseClient,
    bucket: str,
    db_prefix: str,
    max_logs: int | None = None,
) -> list[S3LogObject]:
    """Purpose: Enumerate nuPlan .db objects under a prefix.
    Parameters:
        s3_client (BaseClient): boto3 S3 client.
        bucket (str): Source bucket name.
        db_prefix (str): Prefix containing .db objects.
        max_logs (int | None): Optional cap on returned logs.
    Returns:
        list[S3LogObject]: Ordered list of discovered log objects.
    Called by: pipeline/retrieve_scene_windows_s3.py.
    Calls: s3_client.get_paginator(), paginator.paginate().
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    discovered: list[S3LogObject] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=db_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".db"):
                continue
            log_id = os.path.basename(key).replace(".db", "")
            discovered.append(S3LogObject(bucket=bucket, key=key, log_id=log_id))
            if max_logs is not None and len(discovered) >= max_logs:
                return discovered
    return discovered


def batched_logs(logs: list[S3LogObject], logs_per_batch: int) -> Iterator[list[S3LogObject]]:
    """Purpose: Yield discovered log objects in fixed-size batches.
    Parameters:
        logs (list[S3LogObject]): Full discovered object list.
        logs_per_batch (int): Batch size, must be >= 1.
    Returns:
        Iterator[list[S3LogObject]]: Sequential list batches.
    Called by: pipeline/retrieve_scene_windows_s3.py.
    Calls: None.
    """
    if logs_per_batch < 1:
        raise ValueError("logs_per_batch must be >= 1")
    for idx in range(0, len(logs), logs_per_batch):
        yield logs[idx : idx + logs_per_batch]


def download_db_to_tempfile(s3_client: BaseClient, log_obj: S3LogObject) -> str:
    """Purpose: Download one .db object from S3 to a temp file.
    Parameters:
        s3_client (BaseClient): boto3 S3 client.
        log_obj (S3LogObject): Source S3 object descriptor.
    Returns:
        str: Absolute local path of downloaded temp .db file.
    Called by: pipeline/retrieve_scene_windows_s3.py.
    Calls: tempfile.NamedTemporaryFile(), s3_client.download_file().
    """
    with tempfile.NamedTemporaryFile(prefix="nuplan_", suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    s3_client.download_file(log_obj.bucket, log_obj.key, tmp_path)
    return tmp_path

