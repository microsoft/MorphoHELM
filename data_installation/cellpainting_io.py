"""Small IO helpers for public Cell Painting dataset downloads."""

from __future__ import annotations

import os
import shutil
import tarfile
import time
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable, TypeVar

import boto3
from botocore import UNSIGNED
from botocore.config import Config

T = TypeVar("T")


def unsigned_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def retry(operation: Callable[[], T], *, attempts: int = 5, sleep_seconds: float = 1.0) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if attempt == attempts:
                break
            time.sleep(sleep_seconds * attempt)
    assert last_error is not None
    raise last_error


def read_s3_bytes(uri: str, *, client=None, attempts: int = 5) -> bytes:
    bucket, key = parse_s3_uri(uri)
    s3 = client or unsigned_s3_client()

    def _read() -> bytes:
        response = s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    return retry(_read, attempts=attempts)


def download_url(
    url: str,
    output_path: Path,
    *,
    overwrite: bool = False,
    attempts: int = 8,
    timeout_seconds: int = 300,
) -> Path:
    """Download URL to output_path using an atomic temporary file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return output_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    def _download() -> Path:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        os.replace(tmp_path, output_path)
        return output_path

    try:
        return retry(_download, attempts=attempts)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def extract_tar_members(tar_path: Path, output_dir: Path, *, member_token: str | None = None) -> None:
    """Extract tar members, optionally filtering to names containing member_token."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        members = [m for m in tar.getmembers() if member_token is None or member_token in m.name]
        tar.extractall(output_dir, members=members, filter="data")


def read_zip_member(zip_path: Path, member_name: str) -> bytes:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(member_name)


def list_zip_files(zip_path: Path, suffix: str = "") -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return sorted(name for name in archive.namelist() if not name.endswith("/") and name.endswith(suffix))


def bytes_to_filelike(payload: bytes) -> BytesIO:
    return BytesIO(payload)
