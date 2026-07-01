#!/usr/bin/env python3
"""
collect_training_from_video.py — Extract fish crops from video files for retraining.

Uses YOLO for detection and the classification bundle for species prediction.
Every N seconds a frame is sampled; each detected fish is cropped and saved.

Output layout
-------------
<output_dir>/
  images/
    <uuid>.png          # bbox crop at full resolution
  annotation.json       # training-compatible records (label + segmentation polygon)
  manifest.csv          # human-readable summary with confidence scores
  review/
    <species>/          # crops sorted by predicted species — review and correct before training

Usage
-----
Selection of files - etter utsett
find /data/P-Prosjekter/18200300_overvaking_av_laks_i_tanavassdraget/ARIS-data/2026/Anarjohka_video \
  -type f \
  -name "*.mp4" \
  -newermt "2026-0506-15" \
  > video_list.txt

#per camera YOLO threshold 
python scripts/collect_training_from_video.py \
    --videos $(cat video_list.txt) \
    --det-model Tana_v0.2.pt \
    --cls-model classification_model/model.ts \
    --cls-gallery classification_model/gallery_full_side.pt \
    --cls-method natural_centroid \
    --output /data/P-Prosjekter/18200300_overvaking_av_laks_i_tanavassdraget/ARIS-data/2026/Anarjohka_video/training_crops \
    --frame-interval 1.0 \
    --burst-seconds 3 \
    --conf-threshold 0.0 \
    --det-threshold 0.35 \
    --top-k 3

Tip: use --conf-threshold 0.0 to capture all detections regardless of classification
confidence, then manually correct labels in the review/ folders before retraining.
frame-interval: sjekker en frame per sekund.
burst-seconds: ved deteksjon så går den igjennom alle frames 3 sekunder før og etter.
conf-treshold: artsgjenkjenning
top-k: antall artsforslag som lagres
camera-det-threshold: objektgjenkjenning (fisk/ikke fisk)

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import uuid
from pathlib import Path

# Ensure repo root is on sys.path regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract fish crops from video files for classifier retraining."
    )
    p.add_argument(
        "--videos", nargs="+", required=True,
        help="One or more video paths (MP4, AVI, …).",
    )
    p.add_argument(
        "--det-model", default="model.pt",
        help="Path to the YOLO detection model (.pt). Default: model.pt",
    )
    cls_group = p.add_mutually_exclusive_group(required=True)
    cls_group.add_argument(
        "--cls-bundle",
        help="Self-contained classification bundle (.pt) with embedded centroids.",
    )
    cls_group.add_argument(
        "--cls-model",
        help="Classification TorchScript checkpoint (.ts / .pt).",
    )
    p.add_argument(
        "--cls-gallery", default=None,
        help="Natural-gallery .pt file (required when --cls-model is used).",
    )
    p.add_argument(
        "--output", default="output/training_crops",
        help="Root output directory (default: output/training_crops).",
    )
    p.add_argument(
        "--frame-interval", type=float, default=1.0,
        help="Seconds between sampled frames (default: 1.0). Use 0 for every frame.",
    )
    p.add_argument(
        "--burst-seconds", type=float, default=3.0,
        help="On detection, process every frame this many seconds before and after (default: 3.0).",
    )
    p.add_argument(
        "--conf-threshold", type=float, default=0.0,
        help="Minimum classification confidence to save a crop (default: 0.0 — save all detections).",
    )
    p.add_argument(
        "--det-threshold", type=float, default=0.25,
        help="YOLO detection confidence threshold (default: 0.25).",
    )
    p.add_argument(
        "--camera-det-threshold", nargs="+", default=[],
        metavar="CAM:THRESH",
        help="Per-camera detection threshold overrides, e.g. --camera-det-threshold 1:0.5 3:0.4",
    )
    p.add_argument(
        "--top-k", type=int, default=3,
        help="Number of top classification predictions to record (default: 3).",
    )
    p.add_argument(
        "--min-crop-size", type=int, default=80,
        help="Minimum width AND height (px) of a crop to be saved (default: 80).",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Torch device: 'cpu' or 'cuda' (default: cpu).",
    )
    p.add_argument(
        "--input-size", nargs=2, type=int, metavar=("H", "W"), default=[154, 434],
        help="Classification model input size H W (default: 154 434).",
    )
    p.add_argument(
        "--cls-method",
        choices=["natural_centroid", "arcface_logits", "arcface_centroid"],
        default="natural_centroid",
        help="Classification scoring method (default: natural_centroid).",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Save annotated frames with bboxes to <output>/debug/.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args: argparse.Namespace):
    from ultralytics import YOLO
    from module.classification_package.fish_inference import FishInferenceEngine, InferenceConfig

    det_model = YOLO(args.det_model)

    cfg = InferenceConfig(max_unique_classes=args.top_k)
    if args.cls_bundle:
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
        cls_engine = FishInferenceEngine.from_checkpoint(
            args.cls_model,
            natural_gallery_path=args.cls_gallery,
            input_size=tuple(args.input_size),
            device=args.device,
            config=cfg,
        )

    return det_model, cls_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def polygon_area(poly_norm: list) -> float:
    x = np.array([p[0] for p in poly_norm])
    y = np.array([p[1] for p in poly_norm])
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def parse_video_filename(video_path: Path) -> tuple[str | None, str | None]:
    """
    Parse datetime and camera number from filenames like 20250702_125634_2.mp4.
    Returns (iso_datetime_str, camera_id) or (None, None) if parsing fails.
    """
    from datetime import datetime
    stem = video_path.stem  # e.g. "20250702_125634_2"
    parts = stem.split("_")
    try:
        dt = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S")
        camera_id = parts[2] if len(parts) > 2 else "unknown"
        return dt.strftime("%Y-%m-%dT%H:%M:%S"), camera_id
    except (ValueError, IndexError):
        return None, None


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_video(video_path: Path, args, det_model, cls_engine,
                  frames_dir: Path, detection_dir: Path, review_dir: Path,
                  debug_dir: Path | None,
                  annotation_records: list, manifest_rows: list,
                  ann_id_counter: list, camera_det_thresholds: dict = {}):

    video_datetime, camera_id = parse_video_filename(video_path)
    if video_datetime:
        print(f"  datetime={video_datetime}  camera={camera_id}")
    else:
        print(f"  (could not parse datetime/camera from filename)")
        camera_id = "unknown"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 7.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(round(fps * args.frame_interval))) if args.frame_interval > 0 else 1
    det_threshold = camera_det_thresholds.get(str(camera_id), args.det_threshold)
    if str(camera_id) in camera_det_thresholds:
        print(f"  Using detection threshold {det_threshold} for camera {camera_id}")

    burst_frames = int(round(fps * args.burst_seconds))  # frames in each direction
    # Ring buffer: stores (frame_idx, frame_bgr) for the last burst_frames frames
    from collections import deque
    frame_buffer = deque(maxlen=burst_frames)
    burst_until_idx = -1   # process every frame until this index
    processed_idxs = set() # avoid reprocessing buffered frames already done normally

    # CAP_PROP_FRAME_COUNT is often inaccurate for these videos — don't show
    # a total so the bar doesn't appear to go past 100%
    pbar = tqdm(desc=video_path.name, unit="frame")
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        in_burst = frame_idx <= burst_until_idx
        # Outside burst: sample at normal interval
        if not in_burst and frame_idx % frame_step != 0:
            # Still buffer this frame for potential backward burst
            frame_buffer.append((frame_idx, frame_bgr.copy()))
            frame_idx += 1
            continue

        if frame_idx in processed_idxs:
            frame_idx += 1
            continue
        processed_idxs.add(frame_idx)

        timestamp_s = frame_idx / fps
        img_h, img_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- YOLO detection ---
        try:
            results = det_model(frame_bgr, conf=det_threshold, verbose=False)
        except Exception as e:
            logger.warning(f"Detection failed at t={timestamp_s:.1f}s: {e}")
            frame_idx += 1
            pbar.update(1)
            continue

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            frame_buffer.append((frame_idx, frame_bgr.copy()))
            frame_idx += 1
            pbar.update(1)
            continue

        bboxes_xyxy = boxes.xyxy.cpu().numpy().astype(int)
        det_confs = boxes.conf.cpu().numpy()

        # Detection found — trigger burst mode
        if frame_idx > burst_until_idx:
            # Process all buffered frames (the 3 seconds before this detection)
            for buf_idx, buf_frame in frame_buffer:
                if buf_idx not in processed_idxs:
                    processed_idxs.add(buf_idx)
                    _process_frame(buf_idx, buf_frame, fps, video_datetime, camera_id,
                                   args, det_model, det_threshold, cls_engine,
                                   frames_dir, detection_dir, review_dir, debug_dir,
                                   annotation_records, manifest_rows, ann_id_counter)
                    pbar.update(1)
            frame_buffer.clear()
            burst_until_idx = frame_idx + burst_frames

        # Compute the wall-clock datetime for this frame
        from datetime import datetime, timedelta
        if video_datetime:
            frame_dt = datetime.fromisoformat(video_datetime) + timedelta(seconds=timestamp_s)
            frame_dt_str = frame_dt.strftime("%Y-%m-%dT%H:%M:%S")
            frame_dt_file = frame_dt.strftime("%Y%m%d_%H%M%S")
        else:
            frame_dt_str = None
            frame_dt_file = f"t{timestamp_s:.1f}s"

        debug_frame = frame_bgr.copy() if debug_dir else None
        any_saved = False

        for fish_idx, (x1, y1, x2, y2) in enumerate(bboxes_xyxy):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
            w_crop, h_crop = x2 - x1, y2 - y1

            if debug_frame is not None:
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if w_crop < args.min_crop_size or h_crop < args.min_crop_size:
                continue

            bbox_crop_rgb = frame_rgb[y1:y2, x1:x2].copy()
            bbox_crop_bgr = cv2.cvtColor(bbox_crop_rgb, cv2.COLOR_RGB2BGR)

            # --- Classification ---
            try:
                result = cls_engine.predict(
                    bbox_crop_rgb,
                    bboxes=[[0, 0, w_crop, h_crop]],
                    method=args.cls_method,
                )
                top_preds = result.top_k[: args.top_k]
            except Exception as e:
                logger.warning(f"Classification failed at t={timestamp_s:.1f}s fish{fish_idx}: {e}")
                continue

            if not top_preds:
                continue

            best = top_preds[0]
            if best.accuracy < args.conf_threshold:
                continue

            # --- Save crop into review/<species>/ ---
            species_safe = best.name.replace(" ", "_")
            species_dir = review_dir / species_safe
            species_dir.mkdir(parents=True, exist_ok=True)
            crop_filename = f"cam{camera_id}_{frame_dt_file}_fish{fish_idx:02d}_{best.accuracy:.2f}.png"
            cv2.imwrite(str(species_dir / crop_filename), bbox_crop_bgr)
            any_saved = True

            if debug_frame is not None:
                label = f"cam{camera_id} {best.name} ({best.accuracy:.2f})"
                cv2.putText(debug_frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Annotation record
            crop_seg = [(0, 0), (w_crop, 0), (w_crop, h_crop), (0, h_crop)]
            ann_id_counter[0] += 1
            record = {
                "id": ann_id_counter[0],
                "image_id": crop_filename,
                "segmentation": crop_seg,
                "label": best.name,
                "id_internal": ann_id_counter[0],
                "include_in_odm": False,
                "confidence": round(best.accuracy, 4),
                "source_video": str(video_path),
                "camera_id": camera_id,
                "video_datetime": video_datetime,
                "frame_datetime": frame_dt_str,
                "timestamp_s": round(timestamp_s, 3),
                "frame_idx": frame_idx,
                "crop_file": str(Path("review") / species_safe / crop_filename),
                "crop_width": w_crop,
                "crop_height": h_crop,
            }
            annotation_records.append(record)

            manifest_row = {
                "crop_file": str(Path("review") / species_safe / crop_filename),
                "source_video": video_path.name,
                "camera_id": camera_id,
                "frame_datetime": frame_dt_str,
                "timestamp_s": round(timestamp_s, 3),
                "fish_index": fish_idx,
                "label": best.name,
                "confidence": round(best.accuracy, 4),
                "crop_width": w_crop,
                "crop_height": h_crop,
            }
            for rank, pred in enumerate(top_preds[1:], start=2):
                manifest_row[f"top{rank}_label"] = pred.name
                manifest_row[f"top{rank}_confidence"] = round(pred.accuracy, 4)
            manifest_rows.append(manifest_row)

        # Save full frame with bboxes whenever fish are detected
        if any_saved and len(bboxes_xyxy) > 0:
            max_conf = float(det_confs.max()) if len(det_confs) > 0 else 0.0
            frame_name = f"cam{camera_id}_{frame_dt_file}_det{max_conf:.2f}.jpg"
            # Clean frame (no boxes) for object detection training
            cv2.imwrite(str(detection_dir / frame_name), frame_bgr)
            # YOLO label file alongside clean frame
            label_path = detection_dir / (Path(frame_name).stem + ".txt")
            with open(label_path, "w") as lf:
                for (fx1, fy1, fx2, fy2) in bboxes_xyxy:
                    cx = ((fx1 + fx2) / 2) / img_w
                    cy = ((fy1 + fy2) / 2) / img_h
                    bw = (fx2 - fx1) / img_w
                    bh = (fy2 - fy1) / img_h
                    lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            # Frame with boxes drawn for visual review
            annotated = frame_bgr.copy()
            for (fx1, fy1, fx2, fy2) in bboxes_xyxy:
                cv2.rectangle(annotated, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 200, 255), 2)
            cv2.imwrite(str(frames_dir / frame_name), annotated)

        # Save debug frame (extra annotation overlay)
        if debug_dir and debug_frame is not None and len(bboxes_xyxy) > 0:
            debug_name = f"cam{camera_id}_{frame_dt_file}.jpg"
            cv2.imwrite(str(debug_dir / debug_name), debug_frame)

        frame_buffer.append((frame_idx, frame_bgr.copy()))
        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()


def _process_frame(frame_idx, frame_bgr, fps, video_datetime, camera_id,
                   args, det_model, det_threshold, cls_engine,
                   frames_dir, detection_dir, review_dir, debug_dir,
                   annotation_records, manifest_rows, ann_id_counter):
    """Run detection + classification on a single buffered frame (burst mode)."""
    from datetime import datetime, timedelta

    timestamp_s = frame_idx / fps
    img_h, img_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        results = det_model(frame_bgr, conf=det_threshold, verbose=False)
    except Exception as e:
        logger.warning(f"[burst] Detection failed at t={timestamp_s:.1f}s: {e}")
        return

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return

    bboxes_xyxy = boxes.xyxy.cpu().numpy().astype(int)
    det_confs = boxes.conf.cpu().numpy()

    if video_datetime:
        frame_dt = datetime.fromisoformat(video_datetime) + timedelta(seconds=timestamp_s)
        frame_dt_str = frame_dt.strftime("%Y-%m-%dT%H:%M:%S")
        frame_dt_file = frame_dt.strftime("%Y%m%d_%H%M%S")
    else:
        frame_dt_str = None
        frame_dt_file = f"t{timestamp_s:.1f}s"

    any_saved = False

    for fish_idx, (x1, y1, x2, y2) in enumerate(bboxes_xyxy):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
        w_crop, h_crop = x2 - x1, y2 - y1
        if w_crop < args.min_crop_size or h_crop < args.min_crop_size:
            continue

        bbox_crop_rgb = frame_rgb[y1:y2, x1:x2].copy()
        bbox_crop_bgr = cv2.cvtColor(bbox_crop_rgb, cv2.COLOR_RGB2BGR)

        try:
            result = cls_engine.predict(bbox_crop_rgb, bboxes=[[0, 0, w_crop, h_crop]], method=args.cls_method)
            top_preds = result.top_k[: args.top_k]
        except Exception as e:
            logger.warning(f"[burst] Classification failed at t={timestamp_s:.1f}s: {e}")
            continue

        if not top_preds:
            continue
        best = top_preds[0]
        if best.accuracy < args.conf_threshold:
            continue

        species_safe = best.name.replace(" ", "_")
        species_dir = review_dir / species_safe
        species_dir.mkdir(parents=True, exist_ok=True)
        crop_filename = f"cam{camera_id}_{frame_dt_file}_fish{fish_idx:02d}_{best.accuracy:.2f}.png"
        cv2.imwrite(str(species_dir / crop_filename), bbox_crop_bgr)
        any_saved = True

        crop_seg = [(0, 0), (w_crop, 0), (w_crop, h_crop), (0, h_crop)]
        ann_id_counter[0] += 1
        annotation_records.append({
            "id": ann_id_counter[0],
            "image_id": crop_filename,
            "segmentation": crop_seg,
            "label": best.name,
            "id_internal": ann_id_counter[0],
            "include_in_odm": False,
            "confidence": round(best.accuracy, 4),
            "source_video": "",
            "camera_id": camera_id,
            "video_datetime": video_datetime,
            "frame_datetime": frame_dt_str,
            "timestamp_s": round(timestamp_s, 3),
            "frame_idx": frame_idx,
            "crop_file": str(Path("review") / species_safe / crop_filename),
            "crop_width": w_crop,
            "crop_height": h_crop,
        })
        row = {
            "crop_file": str(Path("review") / species_safe / crop_filename),
            "source_video": "",
            "camera_id": camera_id,
            "frame_datetime": frame_dt_str,
            "timestamp_s": round(timestamp_s, 3),
            "fish_index": fish_idx,
            "label": best.name,
            "confidence": round(best.accuracy, 4),
            "crop_width": w_crop,
            "crop_height": h_crop,
        }
        for rank, pred in enumerate(top_preds[1:], start=2):
            row[f"top{rank}_label"] = pred.name
            row[f"top{rank}_confidence"] = round(pred.accuracy, 4)
        manifest_rows.append(row)

    if any_saved:
        max_conf = float(det_confs.max()) if len(det_confs) > 0 else 0.0
        frame_name = f"cam{camera_id}_{frame_dt_file}_det{max_conf:.2f}.jpg"
        cv2.imwrite(str(detection_dir / frame_name), frame_bgr)
        label_path = detection_dir / (Path(frame_name).stem + ".txt")
        with open(label_path, "w") as lf:
            for (fx1, fy1, fx2, fy2) in bboxes_xyxy:
                cx = ((fx1 + fx2) / 2) / img_w
                cy = ((fy1 + fy2) / 2) / img_h
                bw = (fx2 - fx1) / img_w
                bh = (fy2 - fy1) / img_h
                lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        annotated = frame_bgr.copy()
        for (fx1, fy1, fx2, fy2) in bboxes_xyxy:
            cv2.rectangle(annotated, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 200, 255), 2)
        cv2.imwrite(str(frames_dir / frame_name), annotated)


def print_summary(annotation_records: list):
    from collections import Counter
    counts = Counter(r["label"] for r in annotation_records)
    print("\nSpecies breakdown:")
    for species, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {count:>5}  {species}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    det_model, cls_engine = load_models(args)

    # Parse per-camera detection threshold overrides: "1:0.5" → {"1": 0.5}
    camera_det_thresholds = {}
    for item in args.camera_det_threshold:
        try:
            cam, thresh = item.split(":")
            camera_det_thresholds[cam.strip()] = float(thresh.strip())
        except ValueError:
            print(f"⚠  Ignoring invalid --camera-det-threshold value: {item!r}  (expected format cam:thresh)")
    if camera_det_thresholds:
        print(f"Per-camera detection thresholds: {camera_det_thresholds}")

    out_root = Path(args.output)
    frames_dir = out_root / "frames"
    detection_dir = out_root / "detection_frames"
    review_dir = out_root / "review"
    debug_dir = out_root / "debug" if args.debug else None
    frames_dir.mkdir(parents=True, exist_ok=True)
    detection_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Load any existing annotations so multiple runs accumulate into the same output folder
    annotation_path = out_root / "annotation.json"
    annotation_records: list = []
    if annotation_path.exists():
        with open(annotation_path) as f:
            annotation_records = json.load(f)
        print(f"Resuming: loaded {len(annotation_records)} existing records from {annotation_path}")

    manifest_rows: list = []
    ann_id_counter = [max((r["id"] for r in annotation_records), default=0)]

    # Progress file tracks completed videos independently of annotation records.
    # This ensures videos with zero detections are also skipped on resume.
    progress_path = out_root / "progress.json"
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                completed_videos = set(json.load(f))
            print(f"Resuming: {len(completed_videos)} video(s) already done — skipping.")
        except Exception:
            completed_videos = set()
    else:
        completed_videos = set()

    for video_path_str in args.videos:
        video_path = Path(video_path_str)
        if not video_path.exists():
            logger.warning(f"Video not found, skipping: {video_path}")
            continue
        if str(video_path) in completed_videos:
            print(f"  ⏭  {video_path.name}")
            continue
        print(f"\n▶ {video_path.name}")
        process_video(video_path, args, det_model, cls_engine,
                      frames_dir, detection_dir, review_dir, debug_dir,
                      annotation_records, manifest_rows, ann_id_counter,
                      camera_det_thresholds)

        # Mark done and persist after every video
        completed_videos.add(str(video_path))
        with open(progress_path, "w") as f:
            json.dump(sorted(completed_videos), f, indent=2)
        with open(annotation_path, "w") as f:
            json.dump(annotation_records, f, indent=2)
        if manifest_rows:
            manifest_path = out_root / "manifest.csv"
            fieldnames = list(manifest_rows[0].keys())
            with open(manifest_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(manifest_rows)
        print(f"  💾 {len(annotation_records)} crops saved so far")

    # Write outputs
    with open(annotation_path, "w") as f:
        json.dump(annotation_records, f, indent=2)
    print(f"\n✅ annotation.json  → {annotation_path}  ({len(annotation_records)} total crops)")

    manifest_path = out_root / "manifest.csv"
    if manifest_rows:
        fieldnames = list(manifest_rows[0].keys())
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(manifest_rows)
    print(f"✅ manifest.csv     → {manifest_path}")
    print(f"✅ review crops     → {review_dir}/  (one folder per species)")

    print_summary(annotation_records)


if __name__ == "__main__":
    main()
