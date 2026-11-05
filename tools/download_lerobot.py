"""Download large LeRobot datasets chunk by chunk to avoid timeout.

snapshot_download fails on repos with 200K+ files because the full tree listing
triggers rate limits. This script lists one subfolder at a time via HfApi, then
downloads files with concurrent threads.

Usage:
    python tools/download_lerobot.py \
        --repo IPEC-COMMUNITY/fractal20220817_data_lerobot \
        --local_dir playground/Datasets/OXE_LEROBOT/fractal20220817_data_lerobot
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download


def _download_one(repo_id, filepath, local_dir, max_retries=5):
    """Download a single file with retries."""
    local_path = os.path.join(local_dir, filepath)
    if os.path.exists(local_path):
        return True
    for attempt in range(max_retries):
        try:
            hf_hub_download(
                repo_id, filename=filepath, repo_type="dataset", local_dir=local_dir
            )
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(min(2 ** (attempt + 1), 30))
            else:
                print(f"    FAILED {filepath}: {e}")
                return False


def download_folder(api, repo_id, local_dir, path_in_repo, workers=8, max_retries=5):
    """Download all files under a specific path in the repo using threads."""
    for attempt in range(max_retries):
        try:
            items = list(api.list_repo_tree(
                repo_id, repo_type="dataset", path_in_repo=path_in_repo, recursive=True
            ))
            break
        except Exception as e:
            print(f"  list attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 30))
            else:
                return False

    files = [item.path for item in items if hasattr(item, 'size')]
    if not files:
        return True

    # Filter out already downloaded
    to_download = [f for f in files if not os.path.exists(os.path.join(local_dir, f))]
    if not to_download:
        print(f"  {len(files)}/{len(files)} already done")
        return True

    done = len(files) - len(to_download)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, repo_id, f, local_dir): f for f in to_download}
        for future in as_completed(futures):
            if future.result():
                done += 1
    print(f"  {done}/{len(files)} files")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--skip_data", action="store_true")
    parser.add_argument("--skip_videos", action="store_true")
    parser.add_argument("--start_chunk", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    api = HfApi()

    # Step 1: meta/ and root files
    print("=== Downloading root + meta/ ===")
    # Root files
    try:
        items = list(api.list_repo_tree(args.repo, repo_type="dataset", path_in_repo="", recursive=False))
        for item in items:
            if hasattr(item, 'size'):
                _download_one(args.repo, item.path, args.local_dir)
    except Exception as e:
        print(f"  root listing failed: {e}")

    download_folder(api, args.repo, args.local_dir, "meta", workers=4)

    # Get chunk count
    info_path = os.path.join(args.local_dir, "meta", "info.json")
    info = json.load(open(info_path))
    total_chunks = info.get("total_chunks", 0)
    print(f"Total chunks: {total_chunks}")

    # Step 2: data/ chunk by chunk
    if not args.skip_data:
        print("=== Downloading data/ ===")
        for chunk_idx in range(args.start_chunk, total_chunks):
            chunk_path = f"data/chunk-{chunk_idx:03d}"
            local_chunk = os.path.join(args.local_dir, chunk_path)
            if os.path.isdir(local_chunk) and len(os.listdir(local_chunk)) >= 1000:
                continue
            print(f"  {chunk_path}/ ({chunk_idx+1}/{total_chunks})")
            download_folder(api, args.repo, args.local_dir, chunk_path, workers=args.workers)

    # Step 3: videos/ chunk by chunk
    if not args.skip_videos:
        print("=== Downloading videos/ ===")
        for chunk_idx in range(args.start_chunk, total_chunks):
            chunk_path = f"videos/chunk-{chunk_idx:03d}"
            local_chunk = os.path.join(args.local_dir, chunk_path)
            if os.path.isdir(local_chunk):
                subfiles = sum(len(f) for _, _, f in os.walk(local_chunk))
                if subfiles > 0:
                    print(f"  {chunk_path}/ ({subfiles} files exist, skipping)")
                    continue
            print(f"  {chunk_path}/ ({chunk_idx+1}/{total_chunks})")
            download_folder(api, args.repo, args.local_dir, chunk_path, workers=args.workers)

    print("=== DONE ===")


if __name__ == "__main__":
    main()
