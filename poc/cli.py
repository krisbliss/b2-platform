"""POC-1 CLI — python -m poc.cli score.

Usage (local file):
    python -m poc.cli score \\
        --image path/to/cert.png \\
        --narrative "My father passed away on March 3rd." \\
        --fields '{"case_id": "ABC-123"}'

Usage (GCS URI):
    python -m poc.cli score \\
        --image gs://your-bucket/cert.jpg \\
        --narrative "My father passed away on March 3rd." \\
        --fields '{"case_id": "ABC-123"}'

Accepts either a local file path or a GCS URI (gs://bucket/blob).
GCS download happens at the boundary — Submission always receives bytes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


async def _resolve_image(image_arg: str) -> bytes:
    """Return image bytes from either a local path or a GCS URI.

    GCS URIs (gs://bucket-name/path/to/blob) are downloaded using the
    google-cloud-storage client. Local paths are read from disk.
    """
    if image_arg.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError:
            print(
                "error: google-cloud-storage is required for GCS URIs. "
                "Install it with: pip install google-cloud-storage",
                file=sys.stderr,
            )
            sys.exit(1)

        # Parse gs://bucket-name/path/to/blob
        without_scheme = image_arg[len("gs://"):]
        bucket_name, _, blob_path = without_scheme.partition("/")

        if not bucket_name or not blob_path:
            print(
                f"error: invalid GCS URI '{image_arg}'. "
                "Expected format: gs://bucket-name/path/to/blob",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            return bucket.blob(blob_path).download_as_bytes()
        except Exception as exc:
            print(f"error: failed to download from GCS: {exc}", file=sys.stderr)
            sys.exit(1)

    # Local file path
    path = Path(image_arg)
    if not path.exists():
        print(f"error: image not found at '{image_arg}'", file=sys.stderr)
        sys.exit(1)
    return path.read_bytes()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poc.cli",
        description="B2 POC-1 reliability scorer CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    score_parser = sub.add_parser("score", help="Score a certificate submission.")
    score_parser.add_argument(
        "--image",
        required=True,
        metavar="PATH_OR_GCS_URI",
        help="Local path or GCS URI (gs://bucket/blob) to the certificate image.",
    )
    score_parser.add_argument(
        "--narrative",
        required=True,
        metavar="TEXT",
        help="Free-text narrative from the claimant.",
    )
    score_parser.add_argument(
        "--fields",
        default="{}",
        metavar="JSON",
        help="JSON object of structured case metadata (default: {}).",
    )
    return parser


async def _run_score(image_path: str, narrative: str, fields_json: str) -> None:
    from poc.models import Submission
    from poc.pipeline import run_pipeline

    try:
        case_fields = json.loads(fields_json)
    except json.JSONDecodeError as exc:
        print(f"error: --fields is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    image_bytes = await _resolve_image(image_path)

    try:
        submission = Submission(
            image=image_bytes,
            narrative=narrative,
            case_fields=case_fields,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    result = await run_pipeline(submission)
    print(result.model_dump_json(indent=2))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "score":
        asyncio.run(_run_score(args.image, args.narrative, args.fields))


if __name__ == "__main__":
    main()
