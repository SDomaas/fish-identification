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
git clone https://github.com/SDomaas/fish-identification.git
cd fish-identification
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Then get the model files (see below).

---

## Getting the Model Files

Model files are not stored in this repository. You need two files:

| File | Size | Purpose |
|---|---|---|
| `model.pt` | ~42 MB | YOLO fish detector |
| `classification_model/model.ts` | ~108 MB | ArcFace species classifier |

### Option A — NINA staff (server access)

Copy from the project server via scp:

```bash
scp <user>@t2lipvdiext01.nina.no:~/fish-identification/model.pt .
scp <user>@t2lipvdiext01.nina.no:~/fish-identification/classification_model/model.ts classification_model/
```

Or rsync if you want everything at once:

```bash
rsync -av --include="*.pt" --include="*.ts" --exclude="venv/" \
    <user>@t2lipvdiext01.nina.no:~/fish-identification/ .
```

### Option B — External users

- **`classification_model/model.ts`**: Download from [Fishial.ai](https://www.fishial.ai). This is their ArcFace backbone model — contact them or check their [GitHub](https://github.com/fishial/fish-identification) for download instructions.
- **`model.pt`**: This detector was fine-tuned on Norwegian river footage and is not publicly available. You can train your own using `scripts/prepare_detector_dataset.py` + standard YOLO training on your own data, or use the base YOLOv8 nano weights (`yolo26n.pt` in the repo) as a starting point.

> The gallery files (`classification_model/gallery_*.pt`) are already included in the repository — no download needed.



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

### `scripts/bbox_editor.py`
Browser-based interactive bounding box editor for YOLO `detection_frames/`. Use this to review, correct, add, or delete bounding boxes on frames and save changes back to `.txt` label files.

```bash
python scripts/bbox_editor.py \
    --images /data/.../training_crops/detection_frames \
    --port 5000
```

Then open **http://localhost:5000** in your browser.

**Controls:**

| Action | How |
|---|---|
| Draw new box | Click and drag on empty area |
| Select box | Click an existing box (turns yellow) |
| Resize box | Drag the yellow corner handles |
| Delete selected box | `Delete` / `Backspace` key or Delete button |
| Clear all boxes | Clear all button |
| Save labels | `Ctrl+S` or Save button — writes `.txt` YOLO label file |
| Navigate images | `←` / `→` arrow keys or Prev/Next buttons |
| Filter images | Dropdown: All / Unlabelled only / Labelled only |

> Labels are saved in YOLO format (`0 cx cy w h` normalised) alongside the image files.

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

> ⚠️ Model files are **not stored in this repository** (too large for GitHub). They are stored on the project server and must be copied manually into the repo folder before running the pipeline.

### YOLO Detector — `model.pt`

| Property | Value |
|---|---|
| Architecture | YOLOv8 nano |
| Input size | 640×640 |
| mAP50 | 0.897 (best epoch 39/50) |
| Training data | Karasjohka + Anarjohka underwater footage |
| Classes | 1 (fish) |
| File size | ~42 MB |

**Server location:** `~/fish-identification/model.pt` (on `t2lipvdiext01`)  
Previous checkpoints: `model_backups/`

---

### ArcFace Classifier — `classification_model/model.ts`

| Property | Value |
|---|---|
| Format | TorchScript |
| Architecture | ArcFace backbone (DinoV2/ViT-based) |
| Input size | 154×434 px (H×W) |
| Embedding dim | 512 |
| File size | ~108 MB |
| Source | [Fishial.ai](https://www.fishial.ai) |

**Server location:** `~/fish-identification/classification_model/model.ts` (on `t2lipvdiext01`)

This model is used as a frozen feature extractor. Species classification is done by comparing embeddings to a **gallery** of per-species centroids — no retraining of this model is needed to add new species.

---

### Galleries — `classification_model/*.pt`

Galleries are small (~6 KB) and **are committed to the repository**.

| File | Species | Notes |
|---|---|---|
| `gallery_full_side.pt` | 7 | Built from clean side-view crops — **recommended** |
| `gallery_partial.pt` | 8 | Includes Perca; partial/mixed views |
| `local_gallery.pt` | 8 | Earlier combined gallery |

Rebuild a gallery any time using `scripts/build_species_gallery.py` after adding new sorted crops to the species library.

**Species library location on server:**  
`/data/P-Prosjekter/18200300_overvaking_av_laks_i_tanavassdraget/ARIS-data/2026/Karasjohka_video/training_crops/Species_library/`

---

## Directory Structure

```
scripts/                  Main pipeline scripts
classification_model/     Classifier model and galleries
module/                   Shared Python modules
train_scripts/            Model training scripts
helper/                   Utility notebooks and tools
archive/                  Legacy scripts (kept for reference)
model_backups/            Old detector checkpoints
```
