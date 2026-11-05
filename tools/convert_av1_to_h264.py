#!/usr/bin/env python3
"""In-place convert AV1 mp4 to H264. Skips files already in H264."""
import subprocess
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_codec(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except:
        return "unknown"

def convert_one(src):
    src = Path(src)
    tmp = src.with_suffix(".h264.tmp.mp4")
    try:
        codec = get_codec(src)
        if codec == "h264":
            return "skip"
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264",
             "-crf", "18", "-preset", "fast", "-an", str(tmp), "-loglevel", "error"],
            capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            os.replace(str(tmp), str(src))
            return "ok"
        else:
            tmp.unlink(missing_ok=True)
            return f"fail: {r.stderr[:100]}"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return f"error: {e}"

def process_dir(dataset_dir, jobs=16):
    dataset_dir = Path(dataset_dir)
    print(f"[{time.strftime('%H:%M:%S')}] Scanning {dataset_dir} ...")

    mp4s = list(dataset_dir.rglob("*.mp4"))
    mp4s = [p for p in mp4s if ".h264.tmp" not in p.name]
    print(f"[{time.strftime('%H:%M:%S')}] Found {len(mp4s)} mp4 files, converting with {jobs} workers...")

    done = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(convert_one, p): p for p in mp4s}
        for fut in as_completed(futures):
            result = fut.result()
            if result == "skip":
                skipped += 1
            elif result == "ok":
                done += 1
            else:
                failed += 1
                if failed <= 5:
                    print(f"  FAIL: {futures[fut].name}: {result}")

            total = done + skipped + failed
            if total % 500 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                remaining = (len(mp4s) - total) / rate if rate > 0 else 0
                print(f"  [{time.strftime('%H:%M:%S')}] {total}/{len(mp4s)} "
                      f"(converted={done}, skipped={skipped}, failed={failed}) "
                      f"ETA: {remaining/60:.0f}min")

    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Done: converted={done}, skipped(h264)={skipped}, failed={failed} in {elapsed/60:.1f}min")

if __name__ == "__main__":
    dirs = sys.argv[1:]
    if not dirs:
        print("Usage: python convert_av1_to_h264_inplace.py <dir1> [dir2] ...")
        sys.exit(1)
    for d in dirs:
        process_dir(d, jobs=16)
