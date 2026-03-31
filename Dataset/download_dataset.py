#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional


VIDEO_URLS = [
    "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00",
    "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01",
]

ANNOTATION_URLS = [
     "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip"
]


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=10,
        connect=10,
        read=10,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def remote_file_size(session: requests.Session, url: str) -> Optional[int]:
    try:
        r = session.head(url, allow_redirects=True, timeout=60)
        if r.status_code >= 400:
            return None
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except Exception:
        return None


def guess_filename_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def download_file(
    session: requests.Session,
    url: str,
    out_path: Path,
    chunk_size: int = 8 * 1024 * 1024,
    max_attempts: int = 20,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    expected_size = remote_file_size(session, url)

    for attempt in range(1, max_attempts + 1):
        existing_size = out_path.stat().st_size if out_path.exists() else 0

        if expected_size is not None and existing_size == expected_size:
            print(f"[skip] already complete: {out_path}")
            return

        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        try:
            with session.get(url, stream=True, headers=headers, timeout=(30, 120)) as r:
                if r.status_code == 416:
                    print(f"[skip] already complete: {out_path}")
                    return

                if r.status_code not in (200, 206):
                    raise RuntimeError(f"Bad status code {r.status_code}")

                # If server ignored range request, restart from scratch
                if existing_size > 0 and r.status_code == 200:
                    print(f"[warn] server ignored resume for {out_path.name}, restarting download")
                    existing_size = 0
                    out_path.unlink(missing_ok=True)

                mode = "ab" if r.status_code == 206 and existing_size > 0 else "wb"
                downloaded = existing_size

                if expected_size is not None:
                    print(f"[download] {out_path.name} | attempt {attempt}/{max_attempts}")
                else:
                    print(f"[download] {out_path.name} | attempt {attempt}/{max_attempts} | size unknown")

                with open(out_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)

                        if expected_size:
                            pct = 100.0 * downloaded / expected_size
                            gb_done = downloaded / (1024 ** 3)
                            gb_total = expected_size / (1024 ** 3)
                            print(
                                f"\r  {gb_done:,.2f} / {gb_total:,.2f} GB ({pct:5.1f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            gb_done = downloaded / (1024 ** 3)
                            print(
                                f"\r  {gb_done:,.2f} GB",
                                end="",
                                flush=True,
                            )

                print()

            final_size = out_path.stat().st_size
            if expected_size is None or final_size == expected_size:
                print(f"[done] {out_path}")
                return

            print(
                f"[retry] size mismatch for {out_path.name}: "
                f"{final_size} != {expected_size}"
            )

        except requests.exceptions.RequestException as e:
            print(f"\n[retry] network error on {out_path.name}: {e}")
        except Exception as e:
            print(f"\n[retry] error on {out_path.name}: {e}")

        sleep_s = min(60, 2 ** attempt)
        print(f"[wait] sleeping {sleep_s}s before retry")
        time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {max_attempts} attempts: {url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("something_something_v2"))
    parser.add_argument("--videos-only", action="store_true")
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()

    root = args.root
    archives_dir = root / "archives"
    annotations_dir = root / "annotations"

    archives_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()

    if not args.labels_only:
        for url in VIDEO_URLS:
            filename = guess_filename_from_url(url)
            out_path = archives_dir / filename
            download_file(session, url, out_path)

    if not args.videos_only:
        for url in ANNOTATION_URLS:
            filename = guess_filename_from_url(url)
            out_path = annotations_dir / filename
            download_file(session, url, out_path, chunk_size=1024 * 1024)

    print("\nFinished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)