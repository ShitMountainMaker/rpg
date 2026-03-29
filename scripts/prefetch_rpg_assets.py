#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import sys
import time

import requests
from requests import exceptions as request_exceptions


README_CATEGORIES = [
    "Sports_and_Outdoors",
    "Beauty",
    "Toys_and_Games",
    "CDs_and_Vinyl",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prefetch RPG datasets and HF sentence encoder for remote experiments."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the RPG repository root.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=README_CATEGORIES,
        help="Amazon categories to prefetch. Defaults to the README reproduction set.",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="sentence-transformers/sentence-t5-base",
        help="HF sentence model to cache locally via hf-mirror.",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=None,
        help="Optional Hugging Face cache dir. Defaults to the environment cache.",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip Amazon raw data downloads.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip Hugging Face model caching.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Maximum retries for raw dataset downloads.",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=8,
        help="Streaming download chunk size in MiB.",
    )
    return parser.parse_args()


def ensure_hf_env():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("USE_TF", "0")


def _remote_file_size(url: str) -> int | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
    except request_exceptions.RequestException:
        return None

    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    return int(content_length)


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, request_exceptions.HTTPError):
        response = exc.response
        if response is None:
            return True
        return response.status_code >= 500 or response.status_code in {408, 429}
    return isinstance(
        exc,
        (
            request_exceptions.ChunkedEncodingError,
            request_exceptions.ConnectionError,
            request_exceptions.Timeout,
            OSError,
        ),
    )


def download_file(url: str, path: Path, force: bool, max_retries: int, chunk_size_mb: int):
    if path.exists() and not force:
        print(f"[skip] {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = Path(f"{path}.part")
    remote_size = _remote_file_size(url)

    if force:
        path.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)

    if part_path.exists() and remote_size is not None and part_path.stat().st_size > remote_size:
        print(f"[reset] partial file larger than remote: {part_path}")
        part_path.unlink()

    if part_path.exists() and remote_size is not None and part_path.stat().st_size == remote_size:
        part_path.replace(path)
        print(f"[saved] {path}")
        return

    print(f"[download] {url}")

    chunk_size = chunk_size_mb * 1024 * 1024
    for attempt in range(max_retries + 1):
        resume_from = part_path.stat().st_size if part_path.exists() else 0
        headers = {}
        mode = "ab" if resume_from > 0 else "wb"
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
            print(f"[resume] {path} from byte {resume_from}")

        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as response:
                if resume_from > 0 and response.status_code == 200:
                    print(f"[restart] server ignored range request for {path}")
                    part_path.unlink(missing_ok=True)
                    resume_from = 0
                    mode = "wb"
                response.raise_for_status()
                with open(part_path, mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)

            if remote_size is not None and part_path.stat().st_size != remote_size:
                raise OSError(
                    f"incomplete download for {path}: {part_path.stat().st_size}/{remote_size} bytes"
                )

            part_path.replace(path)
            print(f"[saved] {path}")
            return
        except Exception as exc:
            if attempt >= max_retries or not _should_retry(exc):
                raise
            wait_seconds = min(60, 2**attempt)
            print(f"[retry {attempt + 1}/{max_retries}] {path.name}: {exc} (sleep {wait_seconds}s)")
            time.sleep(wait_seconds)


def prefetch_amazon_raw(
    repo_root: Path, categories: list[str], force: bool, max_retries: int, chunk_size_mb: int
):
    for category in categories:
        raw_dir = repo_root / "cache" / "AmazonReviews2014" / category / "raw"
        reviews_url = (
            "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
            f"reviews_{category}_5.json.gz"
        )
        meta_url = (
            "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
            f"meta_{category}.json.gz"
        )
        download_file(
            reviews_url,
            raw_dir / f"reviews_{category}_5.json.gz",
            force,
            max_retries,
            chunk_size_mb,
        )
        download_file(
            meta_url,
            raw_dir / f"meta_{category}.json.gz",
            force,
            max_retries,
            chunk_size_mb,
        )


def prefetch_hf_model(model_name: str, hf_cache: Path | None):
    ensure_hf_env()
    from huggingface_hub import snapshot_download

    cache_dir = str(hf_cache) if hf_cache is not None else None
    print(f"[hf] endpoint={os.environ['HF_ENDPOINT']}")
    print(f"[hf] model={model_name}")
    if cache_dir is not None:
        print(f"[hf] cache_dir={cache_dir}")

    snapshot_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        resume_download=True,
    )
    print(f"[hf] cached at {snapshot_path}")


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()

    if not repo_root.exists():
        raise FileNotFoundError(f"Repo root does not exist: {repo_root}")

    if not args.skip_dataset:
        prefetch_amazon_raw(
            repo_root,
            args.categories,
            args.force,
            args.max_retries,
            args.chunk_size_mb,
        )

    if not args.skip_model:
        prefetch_hf_model(args.hf_model, args.hf_cache)

    print("")
    print("Suggested training override for the cached HF encoder:")
    print(
        "python main.py "
        "--category=Sports_and_Outdoors "
        "--sent_emb_model=sentence-transformers/sentence-t5-base "
        "--sent_emb_dim=768 "
        "--sent_emb_pca=128"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
