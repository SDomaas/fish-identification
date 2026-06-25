# Fish Identification — Norwegian River Monitoring

Pipeline for automated fish detection, species classification, and individual tracking from underwater camera footage. Deployed on rivers Karasjohka and Anarjohka (Tana watershed).

---

## Overview

The system uses two models in sequence:

1. **YOLO detector** (`model.pt`) — detects fish bounding boxes in each frame
2. **ArcFace classifier** (`classification_model/model.ts`) — identifies species using embedding similarity against a gallery

Camera footage is ~7 fps colour video at 1080×1920 from fixed underwater cameras. The pipeline is iterative: collect crops → sort manually → retrain detector → rebuild gallery → repeat.

---

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Scripts

### `scripts/collect_training_from_video.py`
Extracts fish crops from video for retraining. Samples one frame per second by default; on detection, also processes all frames in a burst window around it.

```bash
python scripts/collect_training_from_video.py \
    --videos $(cat video_list.txt) \
    --det-model model.pt \
    --cls-model classification_model/model.ts \
    --cls-gallery classification_model/gallery_full_side.pt \
    --cls-method natural_centroid \
    --output /data/.../training_crops \
    --frame-interval 1.0 \
    --burst-seconds 3 \
    --conf-threshold 0.0 \
    --det-threshold 0.35 \
    --top-k 3
```

**Key options:**
| Flag | Default | Description |
|---|---|---|
| `--frame-interval` | 1.0 | Seconds between sampled frames |
| `--burst-seconds` | 3 | Extra seconds to sample around each detection |
| `--det-threshold` | 0.35 | YOLO confidence threshold |
| `--conf-threshold` | 0.0 | Min species confidence to save crop |
| `--top-k` | 3 | Number of top species predictions to record |
| `--camera-det-threshold` | — | Per-camera override e.g. `1:0.5 3:0.4` |

**Outputs:** `review/<species>/` (crops), `detection_frames/` (clean frames + YOLO labels), `frames/` (annotated), `annotation.json`, `manifest.csv`

---

### `scripts/track_fish.py`
Tracks individual fish across frames using IoU matching. Saves a video clip and best-crop image per track.

```bash
python scripts/track_fish.py \
    --videos $(cat video_list.txt) \
    --det-model model.pt \
    --cls-model classification_model/model.ts \
    --cls-gallery classification_model/gallery_full_side.pt \
    --cls-method natural_centroid \
    --output /data/.../track_output \
    --det-threshold 0.35 \
    --conf-threshold 0.0 \
    --track-max-gap 15 \
    --buffer-seconds 2.0
```

**Key options:**
| Flag | Default | Description |
|---|---|---|
| `--track-max-gap` | 15 | Frames a fish can disappear before track closes |
| `--buffer-seconds` | 2.0 | Seconds of video padding before/after each track |
| `--location` | auto | River name in output filenames (auto-detected from path) |

**Outputs:** `<species>/<species>_<date>_<time>_cam<N>_<location>_t<NNNN>.mp4` + `.png`

---

### `scripts/build_species_gallery.py`
Builds an ArcFace centroid gallery from manually sorted crop folders.

```bash
python scripts/build_species_gallery.py \
    --review /data/.../species_library/review_full_side \
    --model classification_model/model.ts \
    --output classification_model/gallery_full_side.pt \
    --min-crops 10
```

The gallery is a `.pt` file with one centroid per species. Flip augmentation is applied by default for orientation invariance.

---

### `scripts/prepare_detector_dataset.py`
Builds a YOLO train/val dataset from manually sorted `detection_frames/` folders. Fish frames use saved `.txt` labels; background frames get empty labels.

```bash
python scripts/prepare_detector_dataset.py \
    --detection-frames /data/.../training_crops/detection_frames \
    --output /data/.../yolo_dataset \
    --det-model model.pt
```

Supports multiple `--detection-frames` sources to combine data from different rivers.

---

### `scripts/mirror_sort_to_detection_frames.py`
After manually sorting crops in `review/`, mirrors that sorting into `detection_frames/` (moves images and `.txt` label files to matching subfolders).

---

## Galleries

| File | Species | Notes |
|---|---|---|
| `classification_model/gallery_full_side.pt` | 7 | Built from clean side-view crops — best quality |
| `classification_model/gallery_partial.pt` | 8 | Includes Perca; partial/mixed views |
| `classification_model/local_gallery.pt` | 8 | Earlier combined gallery |

**Use `gallery_full_side.pt`** for production runs. Switch to `gallery_partial.pt` if Perca detection is needed.

---

## Species

Currently in the gallery:
- *Salmo salar* (Atlantic salmon)
- *Salmo trutta* (Brown trout)
- *Thymallus thymallus* (Grayling)
- *Coregonus lavaretus* (Whitefish)
- *Esox lucius* (Pike)
- *Perca fluviatilis* (Perch) — partial gallery only
- *Fiskand* (Merganser)
- smolt/parr

---

## Iterative Retraining Workflow

```
1. collect_training_from_video.py  →  review/ crops
2. Manually sort crops by species
3. build_species_gallery.py        →  updated gallery
4. prepare_detector_dataset.py     →  YOLO dataset
5. yolo train data=...             →  new model.pt
6. Evaluate and repeat
```

---

## Models

| File | Description |
|---|---|
| `model.pt` | Current YOLO detector (mAP50=0.897, fine-tuned on Karasjohka+Anarjohka) |
| `classification_model/model.ts` | TorchScript ArcFace backbone (108 MB, input 154×434) |
| `model_backups/` | Previous detector checkpoints |

---

## Directory Structure

```
scripts/                  Main pipeline scripts
classification_model/     Classifier model and galleries
module/                   Shared Python modules
train_scripts/            Model training scripts
helper/                   Utility notebooks and tools
archive/                  Legacy scripts (kept for reference)
annotation_data_Seavision_COCO/  Manual COCO annotations
model_backups/            Old detector checkpoints
```
