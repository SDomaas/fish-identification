# Scripts

| Script | Purpose |
|---|---|
| `collect_training_from_video.py` | Extract fish crops from video for training data collection |
| `track_fish.py` | Track individual fish and save per-track clips + best crops |
| `bbox_editor.py` | Interactive browser UI to review and edit YOLO bounding box labels |
| `build_species_gallery.py` | Build ArcFace centroid gallery from sorted crop folders |
| `prepare_detector_dataset.py` | Build YOLO dataset from sorted detection_frames folders |
| `mirror_sort_to_detection_frames.py` | Sync manual sorting from review/ to detection_frames/ |
| `clean_fiftyone_dataset.py` | Clean FiftyOne dataset utility |
| `remove_duplicate_boxes.py` | Remove overlapping bounding boxes |
| `object_detection.sh` | Shell wrapper for YOLO training |
| `classification.sh` | Shell wrapper for classification training |

See the main [README](../README.md) for full usage examples.
