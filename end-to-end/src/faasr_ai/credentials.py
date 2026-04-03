# src/faasr_ai/credentials.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REQUIRED_CREDENTIALS = {
    "GH_PAT": "GitHub Personal Access Token",
    "ANTHROPIC_API_KEY": "Anthropic API Key",
    "S3_AccessKey": "S3 Access Key",
    "S3_SecretKey": "S3 Secret Key",
    "FAASR_S3_ENDPOINT": "S3 Endpoint URL (e.g., https://s3.us-east-1.amazonaws.com)",
    "FAASR_S3_BUCKET": "S3 Bucket Name",
    "FAASR_S3_REGION": "S3 Region (e.g., us-east-1)",
    "FAASR_GH_USERNAME": "GitHub Username",
    "FAASR_ACTION_REPO": "GitHub Actions Repository Name",
}


def ensure_credentials(env_path: Path | str = ".env") -> None:
    """Check if .env exists with all required keys. If any are missing, prompt and append."""
    env_path = Path(env_path)
    if env_path.exists():
        load_dotenv(env_path)

    missing = [k for k in REQUIRED_CREDENTIALS if not os.getenv(k)]
    if not missing:
        return

    print("\nFirst-run setup: Please provide the following credentials.\n")
    values: dict[str, str] = {}
    for key in missing:
        value = input(f"  {REQUIRED_CREDENTIALS[key]} ({key}): ").strip()
        values[key] = value
        os.environ[key] = value

    with open(env_path, "a") as f:
        for k, v in values.items():
            f.write(f"{k}={v}\n")

    load_dotenv(env_path, override=True)
    print("\nCredentials saved to .env\n")
