#!/usr/bin/env python3
"""
track_fish.py — Track individual fish in video and save per-track clips and best crops.

For each detected fish track:
  - Saves a .mp4 clip covering the full track + 2-second buffer before/after
  - Saves the single highest-confidence crop from the track

Output layout
-------------
<output_dir>/
  <species>/
    <species>_<YYYYMMDD>_<HHMMSS>_cam<N>_<location>.mp4    # video clip
    <species>_<YYYYMMDD>_<HHMMSS>_cam<N>_<location>.png    # best crop

Usage
-----
Selection of files - dato og lokasjon
find /data/P-Prosjekter/18200300_overvaking_av_laks_i_tanavassdraget/ARIS-data/2026/Karasjohka_video \
  -type f \
  -name "*.mp4" \
  -newermt "2026-06-02" \
  > video_list.txt

python scripts/track_fish.py \
    --videos $(cat video_list.txt) \
    --det-model Tana_v0.2.pt \
    --cls-model classification_model/model.ts \
    --cls-gallery classification_model/gallery_full_side.pt \
    --cls-method natural_centroid \
    --output /data/P-Prosjekter/18200300_overvaking_av_laks_i_tanavassdraget/ARIS-data/2026/video_track_output \
    --det-threshold 0.35 \
    --conf-threshold 0.0 \
    --track-max-gap 15 \
    --buffer-seconds 2.0

Location is auto-detected from the video path if it contains 'Anarjohka' or 'Karasjohka',
otherwise use --location to specify it.
"""

import argparse
import logging
import sys
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Ensure repo root is on sys.path so 'module' package can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")


# ---------------------------------------------------------------------------
# IoU-based tracker
# ---------------------------------------------------------------------------

class SimpleTracker:
    """Lightweight IoU-based tracker. Links detections frame-to-frame by IoU."""

    def __init__(self, max_gap: int = 15, min_iou: float = 0.25):
        self.max_gap = max_gap
        self.min_iou = min_iou
        self._next_id = 1
        self._active: dict[int, dict] = {}
        self._finished: list[dict] = []

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)

    def update(self, frame_idx: int, detections: list[dict]) -> list[int | None]:
        """Match detections to active tracks. Returns list of track_ids (same order as detections)."""
        for tid in [t for t, s in self._active.items()
                    if frame_idx - s['last_frame'] > self.max_gap]:
            self._finished.append(self._active.pop(tid))

        if not detections:
            return []

        result_ids: list[int] = [-1] * len(detections)
        matched_tracks: set[int] = set()

        for di, det in enumerate(detections):
            best_iou, best_tid = self.min_iou, None
            for tid, state in self._active.items():
                if tid in matched_tracks:
                    continue
                iou = self._iou(det['bbox'], state['last_bbox'])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None:
                result_ids[di] = best_tid
                matched_tracks.add(best_tid)

        for di, det in enumerate(detections):
            tid = result_ids[di]
            if tid == -1:
                tid = self._next_id
                self._next_id += 1
                self._active[tid] = {
                    'track_id': tid,
                    'frames': [],
                    'first_frame': frame_idx,
                    'last_frame': frame_idx,
                    'last_bbox': det['bbox'],
                }
                result_ids[di] = tid
            self._active[tid]['last_frame'] = frame_idx
            self._active[tid]['last_bbox'] = det['bbox']
            self._active[tid]['frames'].append({**det, 'frame_idx': frame_idx})

        return result_ids

    def pop_finished(self) -> list[dict]:
        result, self._finished = self._finished[:], []
        return result

    def finalize_all(self) -> list[dict]:
        for state in self._active.values():
            self._finished.append(state)
        self._active.clear()
        return self.pop_finished()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_video_filename(video_path: Path) -> tuple[str | None, str | None]:
    """Parse datetime and camera ID from filenames like 20260601_232002_3.mp4."""
    stem = video_path.stem
    parts = stem.split("_")
    try:
        dt = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S")
        camera_id = parts[2] if len(parts) > 2 else "unknown"
        return dt.strftime("%Y-%m-%dT%H:%M:%S"), camera_id
    except (ValueError, IndexError):
        return None, None


def detect_location(video_path: Path, fallback: str) -> str:
    """Infer river location from the video path."""
    path_str = str(video_path)
    if "Anarjohka" in path_str:
        return "Anarjohka"
    if "Karasjohka" in path_str:
        return "Karasjohka"
    return fallback


def majority_species(track: dict) -> str:
    """Return the most-common predicted species across all track frames."""
    from collections import Counter
    counts = Counter(f['species'] for f in track['frames'])
    return counts.most_common(1)[0][0]


def best_frame(track: dict) -> dict:
    """Return the highest-confidence detection frame in the track."""
    return max(track['frames'], key=lambda f: f['confidence'])


def build_output_stem(species: str, video_datetime: str | None, camera_id: str,
                      location: str, track_id: int) -> str:
    """Build the base filename stem: species_YYYYMMDD_HHMMSS_camN_location_tNNNN"""
    dt_part = video_datetime.replace("-", "").replace(":", "").replace("T", "_") \
        if video_datetime else "unknown"
    species_safe = species.replace(" ", "_")
    return f"{species_safe}_{dt_part}_cam{camera_id}_{location}_t{track_id:04d}"


# ---------------------------------------------------------------------------
# Per-track clip writer
# ---------------------------------------------------------------------------

def save_track_outputs(track: dict, video_path: Path, fps: float,
                       buffer_seconds: float, min_track_seconds: float,
                       out_dir: Path,
                       video_datetime: str | None, camera_id: str, location: str):
    """
    For a finished track: save best-crop PNG and a .mp4 clip with buffer.
    Tracks shorter than min_track_seconds are discarded as likely debris/noise.
    """
    if not track['frames']:
        return

    track_duration = (track['last_frame'] - track['first_frame']) / fps
    if track_duration < min_track_seconds:
        return

    species = majority_species(track)
    best = best_frame(track)
    species_safe = species.replace(" ", "_")
    species_dir = out_dir / species_safe
    species_dir.mkdir(parents=True, exist_ok=True)

    stem = build_output_stem(species, video_datetime, camera_id, location, track['track_id'])

    # --- Save best crop ---
    cv2.imwrite(str(species_dir / f"{stem}.png"), best['crop_bgr'])

    # --- Save video clip ---
    buffer_frames = int(round(buffer_seconds * fps))
    first_frame = track['first_frame']
    last_frame = track['last_frame']
    clip_start = max(0, first_frame - buffer_frames)
    clip_end = last_frame + buffer_frames  # may be past end of video; handled below

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot re-open video for clip: {video_path}")
        return

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_path = species_dir / f"{stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (img_w, img_h))

    # Read sequentially to clip_start, then write until clip_end
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx > clip_end:
            break
        if frame_idx >= clip_start:
            # Draw bbox overlay for frames that had a detection in this track
            track_frame_map = {f['frame_idx']: f for f in track['frames']}
            if frame_idx in track_frame_map:
                det = track_frame_map[frame_idx]
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['species']} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    logger.info(f"  Saved track {track['track_id']:04d} ({species}): {clip_path.name}")


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(video_path: Path, args, det_model, cls_engine, out_dir: Path):
    video_datetime, camera_id = parse_video_filename(video_path)
    location = detect_location(video_path, args.location)
    if video_datetime:
        print(f"  datetime={video_datetime}  camera={camera_id}  location={location}")
    else:
        print(f"  (could not parse datetime from filename)  camera=unknown  location={location}")
        camera_id = "unknown"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 7.0
    tracker = SimpleTracker(max_gap=args.track_max_gap)

    # Determine detection threshold for this camera
    det_threshold = args.camera_det_thresholds.get(camera_id, args.det_threshold)

    from tqdm import tqdm
    pbar = tqdm(unit="frame", desc=video_path.name)

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        img_h, img_w = frame_bgr.shape[:2]

        try:
            results = det_model(frame_bgr, conf=det_threshold, verbose=False)
        except Exception as e:
            logger.warning(f"Detection failed at frame {frame_idx}: {e}")
            frame_idx += 1
            pbar.update(1)
            continue

        boxes = results[0].boxes
        detections = []
        if boxes is not None and len(boxes) > 0:
            bboxes_xyxy = boxes.xyxy.cpu().numpy().astype(int)

            for fish_idx, (x1, y1, x2, y2) in enumerate(bboxes_xyxy):
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
                w_crop, h_crop = x2 - x1, y2 - y1
                if w_crop < args.min_crop_size or h_crop < args.min_crop_size:
                    continue

                bbox_crop_rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2].copy(), cv2.COLOR_BGR2RGB)
                bbox_crop_bgr = frame_bgr[y1:y2, x1:x2].copy()

                try:
                    result = cls_engine.predict(
                        bbox_crop_rgb,
                        bboxes=[[0, 0, w_crop, h_crop]],
                        method=args.cls_method,
                    )
                    top_preds = result.top_k[:1]
                except Exception as e:
                    logger.warning(f"Classification failed at frame {frame_idx}: {e}")
                    continue

                if not top_preds:
                    continue
                best = top_preds[0]
                if best.accuracy < args.conf_threshold:
                    continue

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'crop_bgr': bbox_crop_bgr,
                    'species': best.name,
                    'confidence': best.accuracy,
                })

        tracker.update(frame_idx, detections)

        # Save outputs for any tracks that just closed
        for finished in tracker.pop_finished():
            save_track_outputs(
                finished, video_path, fps, args.buffer_seconds,
                args.min_track_seconds, out_dir, video_datetime, camera_id, location,
            )

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # Finalize remaining open tracks at end of video
    for finished in tracker.finalize_all():
        save_track_outputs(
            finished, video_path, fps, args.buffer_seconds,
            args.min_track_seconds, out_dir, video_datetime, camera_id, location,
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Track fish in video and save per-track clips + best crops.")
    p.add_argument("--videos", nargs="+", required=True, help="Input video file(s).")
    p.add_argument("--det-model", required=True, help="YOLO detection model (.pt).")
    p.add_argument("--cls-model", default=None, help="TorchScript classification model (.ts).")
    p.add_argument("--cls-bundle", default=None, help="Classification bundle path (alternative to --cls-model).")
    p.add_argument("--cls-gallery", default=None, help="Gallery .pt file for natural_centroid method.")
    p.add_argument("--cls-method",
                   choices=["natural_centroid", "arcface_logits", "arcface_centroid"],
                   default="natural_centroid")
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument("--location", default="unknown",
                   help="River/location label for filenames (auto-detected if path contains Anarjohka/Karasjohka).")
    p.add_argument("--det-threshold", type=float, default=0.35,
                   help="YOLO detection confidence threshold (default: 0.35).")
    p.add_argument("--conf-threshold", type=float, default=0.0,
                   help="Minimum species classification confidence to record (default: 0.0).")
    p.add_argument("--track-max-gap", type=int, default=15,
                   help="Max frames a fish can disappear before its track closes (default: 15).")
    p.add_argument("--min-track-seconds", type=float, default=1.0,
                   help="Minimum track duration in seconds to save (default: 1.0). "
                        "Shorter tracks are discarded as likely debris or false detections.")
    p.add_argument("--buffer-seconds", type=float, default=2.0,
                   help="Seconds of video to include before/after each track (default: 2.0).")
    p.add_argument("--min-crop-size", type=int, default=80,
                   help="Minimum crop width AND height in pixels (default: 80).")
    p.add_argument("--device", default="cpu", help="Torch device: cpu or cuda (default: cpu).")
    p.add_argument("--input-size", nargs=2, type=int, metavar=("H", "W"), default=[154, 434],
                   help="Classification model input size H W (default: 154 434).")
    p.add_argument("--camera-det-threshold", nargs="*", default=[],
                   metavar="CAM:THRESH",
                   help="Per-camera detection threshold overrides, e.g. 1:0.5 3:0.4")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args):
    from ultralytics import YOLO
    from module.classification_package.fish_inference import FishInferenceEngine, InferenceConfig

    det_model = YOLO(args.det_model)
    cfg = InferenceConfig(max_unique_classes=1)

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
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Parse per-camera threshold overrides
    args.camera_det_thresholds = {}
    for item in args.camera_det_threshold:
        try:
            cam, thresh = item.split(":")
            args.camera_det_thresholds[cam.strip()] = float(thresh.strip())
        except ValueError:
            print(f"⚠  Ignoring invalid --camera-det-threshold value: {item!r}")

    det_model, cls_engine = load_models(args)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for video_path_str in args.videos:
        video_path = Path(video_path_str)
        if not video_path.exists():
            logger.warning(f"Video not found, skipping: {video_path}")
            continue
        print(f"\n▶ {video_path.name}")
        process_video(video_path, args, det_model, cls_engine, out_dir)

    print(f"\n✅ Tracks saved to: {out_dir}/  (one subfolder per species)")


if __name__ == "__main__":
    main()
