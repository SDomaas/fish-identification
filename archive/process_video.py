#!/usr/bin/env python3
"""
process_video.py — Run fish detection + classification on an MP4 video.

Pipeline per sampled frame:
  1. Segment fish using SegmentationInference (TorchScript).
  2. Classify each detected crop using FishInferenceEngine (TorchScript bundle
     or checkpoint + natural-gallery pair).
  3. Write results to a CSV with columns: timestamp_s, species, confidence.

Usage examples
--------------
# With a self-contained bundle:
python scripts/process_video.py \
    --video my_video.mp4 \
    --seg-model models/segmentator.ts \
    --cls-bundle models/classification_bundle.pt \
    --output detections.csv

# With a checkpoint + natural gallery pair:
python scripts/process_video.py \
    --video my_video.mp4 \
    --seg-model models/segmentator.ts \
    --cls-model models/classification.ts \
    --cls-gallery models/natural_gallery.pt \
    --output detections.csv \
    --frame-interval 1.0 \
    --top-k 1
"""

import argparse
import csv
import sys
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect and classify fish in an MP4 video."
    )
    p.add_argument("--video", required=True, help="Path to the input MP4 video.")
    p.add_argument(
        "--seg-model",
        required=True,
        help="Path to the segmentation TorchScript model (.ts / .pt).",
    )

    cls_group = p.add_mutually_exclusive_group(required=True)
    cls_group.add_argument(
        "--cls-bundle",
        help="Path to a self-contained classification bundle (.pt) that embeds "
             "centroids and class mapping.",
    )
    cls_group.add_argument(
        "--cls-model",
        help="Path to a classification TorchScript checkpoint (.ts / .pt).",
    )

    p.add_argument(
        "--cls-gallery",
        default=None,
        help="Path to the natural-gallery .pt file (required when --cls-model is used).",
    )
    p.add_argument(
        "--output",
        default="detections.csv",
        help="Output CSV file path (default: detections.csv).",
    )
    p.add_argument(
        "--frame-interval",
        type=float,
        default=1.0,
        help="Seconds between sampled frames (default: 1.0). "
             "Use 0 to process every frame.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions to record per detected fish (default: 1).",
    )
    p.add_argument(
        "--seg-threshold",
        type=float,
        default=0.3,
        help="Segmentation confidence threshold (default: 0.3).",
    )
    p.add_argument(
        "--cls-method",
        choices=["natural_centroid", "arcface_logits", "arcface_centroid"],
        default="natural_centroid",
        help="Classification scoring method (default: natural_centroid).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device: 'cpu' or 'cuda' (default: cpu).",
    )
    p.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=[154, 434],
        help="Classification model input size H W (default: 154 434).",
    )
    return p.parse_args()


def load_models(args):
    """Load segmentation and classification models."""
    # --- segmentation ---
    from module.segmentation_package.interpreter_segm import SegmentationInference

    logger.info(f"Loading segmentation model from {args.seg_model}")
    seg_model = SegmentationInference(args.seg_model)
    seg_model.re_init_model(args.seg_threshold)

    # --- classification ---
    from module.classification_package.fish_inference import (
        FishInferenceEngine,
        InferenceConfig,
    )

    cfg = InferenceConfig(max_unique_classes=args.top_k)

    if args.cls_bundle:
        logger.info(f"Loading classification bundle from {args.cls_bundle}")
        cls_engine = FishInferenceEngine.from_bundle(
            args.cls_bundle,
            input_size=tuple(args.input_size),
            device=args.device,
            config=cfg,
        )
    else:
        if not args.cls_gallery:
            logger.error("--cls-gallery is required when --cls-model is used.")
            sys.exit(1)
        logger.info(f"Loading classification model from {args.cls_model}")
        cls_engine = FishInferenceEngine.from_checkpoint(
            args.cls_model,
            natural_gallery_path=args.cls_gallery,
            input_size=tuple(args.input_size),
            device=args.device,
            config=cfg,
        )

    return seg_model, cls_engine


def extract_bbox_from_polygon(polygon_dict: dict) -> list:
    """Derive [x1, y1, x2, y2] bbox from a segmentation polygon dict."""
    xs = [v for k, v in polygon_dict.items() if k.startswith("x")]
    ys = [v for k, v in polygon_dict.items() if k.startswith("y")]
    if not xs or not ys:
        return []
    return [min(xs), min(ys), max(xs), max(ys)]


def polygon_dict_to_list(polygon_dict: dict) -> list:
    """Convert polygon dict {x1, y1, x2, y2, ...} to [[x,y], ...] list."""
    n = len(polygon_dict) // 2
    return [
        [polygon_dict[f"x{i+1}"], polygon_dict[f"y{i+1}"]]
        for i in range(n)
        if f"x{i+1}" in polygon_dict and f"y{i+1}" in polygon_dict
    ]


def process_video(args, seg_model, cls_engine) -> list[dict]:
    """
    Iterate over sampled frames, run segmentation + classification,
    and collect detection rows.
    """
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps

    frame_step = max(1, int(round(fps * args.frame_interval))) if args.frame_interval > 0 else 1
    sampled_frames = total_frames // frame_step if frame_step > 1 else total_frames

    logger.info(
        f"Video: {args.video}  |  FPS={fps:.1f}  |  "
        f"Duration={duration_s:.1f}s  |  Total frames={total_frames}"
    )
    logger.info(
        f"Sampling every {frame_step} frame(s) → ~{sampled_frames} frames to process"
    )

    rows = []
    frame_idx = 0
    processed = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        timestamp_s = frame_idx / fps
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Segmentation ---
        try:
            polygons, masks = seg_model.inference(frame_rgb)
        except Exception as e:
            logger.warning(f"Segmentation failed at t={timestamp_s:.2f}s: {e}")
            frame_idx += 1
            processed += 1
            continue

        if not polygons:
            frame_idx += 1
            processed += 1
            logger.debug(f"t={timestamp_s:.2f}s — no fish detected")
            continue

        # --- Classification for each detected fish ---
        for fish_idx, (polygon_dict, mask_crop) in enumerate(zip(polygons, masks)):
            # mask_crop is already the cropped fish image (BGR from cv2.bitwise_and)
            # convert to RGB for the classifier
            crop_rgb = cv2.cvtColor(mask_crop, cv2.COLOR_BGR2RGB)

            poly_list = polygon_dict_to_list(polygon_dict)
            bbox = extract_bbox_from_polygon(polygon_dict)

            try:
                result = cls_engine.predict(
                    crop_rgb,
                    bboxes=[bbox] if bbox else None,
                    polys=[poly_list] if poly_list else None,
                    method=args.cls_method,
                )
                top_preds = result.top_k[: args.top_k]
            except Exception as e:
                logger.warning(
                    f"Classification failed for fish {fish_idx} at "
                    f"t={timestamp_s:.2f}s: {e}"
                )
                continue

            for pred in top_preds:
                rows.append(
                    {
                        "timestamp_s": round(timestamp_s, 3),
                        "species": pred.name,
                        "confidence": round(pred.accuracy, 4),
                    }
                )
                logger.info(
                    f"t={timestamp_s:.2f}s | fish {fish_idx+1}/{len(polygons)} | "
                    f"{pred.name} ({pred.accuracy:.2%})"
                )

        frame_idx += 1
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Processed {processed}/{sampled_frames} sampled frames…")

    cap.release()
    logger.info(f"Done. Processed {processed} frames, found {len(rows)} detections.")
    return rows


def write_csv(rows: list[dict], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_s", "species", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Results written to {output_path}  ({len(rows)} rows)")


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("\nNo detections found.")
        return

    header = f"{'Timestamp (s)':>14}  {'Species':<40}  {'Confidence':>10}"
    sep = "-" * len(header)
    print(f"\n{header}\n{sep}")
    for r in rows:
        print(f"{r['timestamp_s']:>14.3f}  {r['species']:<40}  {r['confidence']:>10.4f}")
    print(sep)
    print(f"Total detections: {len(rows)}\n")


def main():
    args = parse_args()
    seg_model, cls_engine = load_models(args)
    rows = process_video(args, seg_model, cls_engine)
    write_csv(rows, args.output)
    print_table(rows)


if __name__ == "__main__":
    main()
