"""
Download the yearly "Historical Rainfall across Singapore" CSV files
(2017-2024) from data.gov.sg.

data.gov.sg exposes a two-step download flow for its tabular datasets:

    GET https://api-open.data.gov.sg/v1/public/api/datasets/{id}/poll-download

returns a JSON body containing a short-lived, signed S3 URL under
``data.url``.  We poll that endpoint until the status is DOWNLOAD_SUCCESS
and then stream the CSV to ``data/raw``.

Usage
-----
    python src/download_data.py                # all years 2017-2024
    python src/download_data.py --years 2023 2024
    python src/download_data.py --force        # re-download even if present
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


def poll_download_url(dataset_id: str, max_tries: int = 10, sleep_s: float = 3.0) -> str:
    """Poll the data.gov.sg download endpoint until a signed URL is ready."""
    url = config.POLL_DOWNLOAD_URL.format(dataset_id=dataset_id)
    last_status = None
    for attempt in range(1, max_tries + 1):
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", {}) or {}
        status = data.get("status")
        last_status = status
        if status == "DOWNLOAD_SUCCESS" and data.get("url"):
            return data["url"]
        # Any other status: wait and retry (the file is being prepared).
        print(f"    poll attempt {attempt}: status={status!r}, retrying...")
        time.sleep(sleep_s)
    raise RuntimeError(
        f"Could not obtain a download URL for {dataset_id} "
        f"(last status={last_status!r})"
    )


def stream_to_file(url: str, dest_path: str) -> int:
    """Stream a (potentially large) file to disk. Returns bytes written."""
    tmp_path = dest_path + ".part"
    total = 0
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
    os.replace(tmp_path, dest_path)
    return total


def download_year(year: int, force: bool = False) -> str:
    dataset_id = config.DATASET_IDS[year]
    dest = os.path.join(config.RAW_DIR, f"rainfall_{year}.csv")
    if os.path.exists(dest) and not force:
        size_mb = os.path.getsize(dest) / 1e6
        print(f"[{year}] already present ({size_mb:.1f} MB) - skipping")
        return dest

    print(f"[{year}] requesting signed download URL ({dataset_id})...")
    signed_url = poll_download_url(dataset_id)
    print(f"[{year}] downloading -> {dest}")
    n = stream_to_file(signed_url, dest)
    print(f"[{year}] done: {n / 1e6:.1f} MB")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SG rainfall CSVs")
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=config.YEARS,
        help="Years to download (default: 2017-2024)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file exists"
    )
    args = parser.parse_args()

    config.ensure_dirs()
    for year in args.years:
        if year not in config.DATASET_IDS:
            print(f"[{year}] no dataset id known - skipping")
            continue
        download_year(year, force=args.force)


if __name__ == "__main__":
    main()
