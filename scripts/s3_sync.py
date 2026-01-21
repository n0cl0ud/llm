#!/usr/bin/env python3
"""
S3 sync utility for training data and adapters.
Pulls Vibe CLI logs from S3 and pushes trained adapters back.
"""

import os
import argparse
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def get_s3_client():
    """Create S3 client using environment credentials."""
    return boto3.client("s3")


def sync_pull(bucket: str, prefix: str, local_dir: str):
    """Pull training data from S3 to local directory."""
    s3 = get_s3_client()
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Pulling from s3://{bucket}/{prefix} to {local_dir}")

    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative_path = key[len(prefix):].lstrip("/")

                if not relative_path:
                    continue

                local_file = local_path / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)

                print(f"  Downloading: {key}")
                s3.download_file(bucket, key, str(local_file))

        print("Pull complete!")

    except ClientError as e:
        print(f"Error pulling from S3: {e}")
        raise


def sync_push(bucket: str, prefix: str, local_dir: str):
    """Push local files to S3."""
    s3 = get_s3_client()
    local_path = Path(local_dir)

    if not local_path.exists():
        print(f"Local directory does not exist: {local_dir}")
        return

    print(f"Pushing from {local_dir} to s3://{bucket}/{prefix}")

    try:
        for local_file in local_path.rglob("*"):
            if local_file.is_file():
                relative_path = local_file.relative_to(local_path)
                s3_key = f"{prefix}/{relative_path}".replace("\\", "/")

                print(f"  Uploading: {s3_key}")
                s3.upload_file(str(local_file), bucket, s3_key)

        print("Push complete!")

    except ClientError as e:
        print(f"Error pushing to S3: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Sync training data with S3")
    parser.add_argument("action", choices=["pull", "push"], help="Sync direction")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET"), help="S3 bucket name")
    parser.add_argument("--data-prefix", default="vibe-logs", help="S3 prefix for training data")
    parser.add_argument("--adapter-prefix", default="adapters", help="S3 prefix for adapters")
    parser.add_argument("--data-dir", default="/data", help="Local data directory")
    parser.add_argument("--adapter-dir", default="/adapters", help="Local adapter directory")
    args = parser.parse_args()

    if not args.bucket:
        print("Error: S3_BUCKET environment variable or --bucket required")
        return

    if args.action == "pull":
        sync_pull(args.bucket, args.data_prefix, args.data_dir)
    elif args.action == "push":
        sync_push(args.bucket, args.adapter_prefix, args.adapter_dir)


if __name__ == "__main__":
    main()
