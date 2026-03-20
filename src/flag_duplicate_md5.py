"""
flag_duplicates.py — Find duplicate images by MD5 hash and add them to bad_images.csv.
Keeps the lowest-numbered file from each duplicate group; flags the rest.
"""

import os
import hashlib
import pandas as pd
from collections import defaultdict

IMG_DIR      = os.path.join("data", "training_data", "training_data")
BAD_IMG_CSV  = os.path.join("data", "bad_images.csv")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("[INFO] Hashing images ...")
    buckets = defaultdict(list)
    for fname in os.listdir(IMG_DIR):
        if fname.lower().endswith(".png"):
            buckets[md5(os.path.join(IMG_DIR, fname))].append(fname)

    duplicates = {h: sorted(fnames) for h, fnames in buckets.items() if len(fnames) > 1}

    if not duplicates:
        print("[INFO] No duplicates found.")
        return

    # Load existing bad list so we don't double-add
    existing = set()
    if os.path.exists(BAD_IMG_CSV):
        existing = set(pd.read_csv(BAD_IMG_CSV)["filename"].astype(str))

    to_flag = []
    for fnames in duplicates.values():
        keep = fnames[0]   # lowest filename (sorted lexically) is kept
        for fname in fnames[1:]:
            if fname not in existing:
                to_flag.append(fname)
        print(f"  keep {keep} | flag {fnames[1:]}")

    if not to_flag:
        print("[INFO] All duplicates already flagged.")
        return

    write_header = not os.path.exists(BAD_IMG_CSV)
    with open(BAD_IMG_CSV, "a") as f:
        if write_header:
            f.write("filename,comment\n")
        for fname in to_flag:
            f.write(f"{fname},duplicate md5\n")

    print(f"[INFO] Flagged {len(to_flag)} duplicate(s) -> {BAD_IMG_CSV}")


if __name__ == "__main__":
    main()