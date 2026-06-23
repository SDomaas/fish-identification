#!/usr/bin/env python3
"""
Mirror the subfolder sorting from 'frames/' into 'detection_frames/'.

After manually sorting frames/ into subfolders (e.g. fisk/, bakgrunn/, etc.),
run this script to move the matching files (image + .txt label) in
detection_frames/ into the same subfolder structure.

Usage:
  python scripts/mirror_sort_to_detection_frames.py \
      --frames /data/P-Prosjekter/18200300_overvaking_av_laks_i_tanavassdraget/ARIS-data/2026/Anarjohka_video/training_crops/frames \
      --detection-frames /data/P-Prosjekter/18200300_overvaking_av_laks_i_tanavassdraget/ARIS-data/2026/Anarjohka_video/training_crops/detection_frames

The script only MOVES files that have been sorted into subfolders.
Files still in the root of detection_frames/ that have no match in frames/
are left untouched.
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True,
                        help="Path to sorted frames/ directory")
    parser.add_argument("--detection-frames", required=True,
                        help="Path to detection_frames/ directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be moved without doing it")
    args = parser.parse_args()

    frames_root = Path(args.frames)
    det_root = Path(args.detection_frames)

    moved = 0
    missing = 0

    # Walk all subfolders in frames/ (skip root-level unsorted files)
    for img_path in sorted(frames_root.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if img_path.parent == frames_root:
            continue  # skip unsorted files still in root

        subfolder = img_path.parent.name
        src_img = det_root / img_path.name
        src_lbl = det_root / (img_path.stem + ".txt")

        if not src_img.exists():
            missing += 1
            continue

        dst_dir = det_root / subfolder
        if not args.dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_img), dst_dir / src_img.name)
            if src_lbl.exists():
                shutil.move(str(src_lbl), dst_dir / src_lbl.name)
        else:
            print(f"  {img_path.name} → {subfolder}/")

        moved += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done!")
    print(f"  Moved:   {moved} frames (+ label files)")
    print(f"  Missing: {missing} (sorted in frames/ but not found in detection_frames/)")

    # Summary of what's in each subfolder
    print("\ndetection_frames/ subfolders:")
    for d in sorted(det_root.iterdir()):
        if d.is_dir():
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            print(f"  {d.name}: {n} images")


if __name__ == "__main__":
    main()
