#!/usr/bin/env python3
"""
Build a local species gallery from crop images for the fish classifier.

Extracts embeddings from labeled crop images using model.ts, computes
per-species centroids, and saves a gallery .pt file that can be used with:

  python scripts/collect_training_from_video.py \
      --cls-model classification_model/model.ts \
      --cls-gallery classification_model/local_gallery.pt \
      --cls-method natural_centroid \
      ...

Usage:
  python scripts/build_species_gallery.py \
      --review /path/to/review \
      --model classification_model/model.ts \
      --output classification_model/local_gallery.pt \
      --min-crops 10

The review/ folder must contain one subfolder per species:
  review/
    Salmo_salar/
    Salmo_trutta/
    ...
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def build_transform(input_size=(154, 434)):
    h, w = input_size
    return transforms.Compose([
        transforms.Resize((h, w), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def extract_embedding(model, img_path: Path, transform, device: str,
                      flip_augment: bool = False) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    imgs = [img]
    if flip_augment:
        imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    embeddings = []
    for im in imgs:
        tensor = transform(im).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            emb = output[0] if isinstance(output, (tuple, list)) else output
        embeddings.append(F.normalize(emb.squeeze(0), p=2, dim=0).cpu())

    return F.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Build local species gallery for classifier")
    parser.add_argument("--review", required=True,
                        help="Path to review/ folder with one subfolder per species")
    parser.add_argument("--model", default="classification_model/model.ts",
                        help="Path to model.ts (TorchScript classifier)")
    parser.add_argument("--output", default="classification_model/local_gallery.pt",
                        help="Output gallery .pt file path")
    parser.add_argument("--min-crops", type=int, default=10,
                        help="Minimum crops required to include a species (default: 10)")
    parser.add_argument("--max-crops", type=int, default=500,
                        help="Max crops per species (randomly sampled if exceeded)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--input-size", nargs=2, type=int, default=[154, 434],
                        metavar=("H", "W"), help="Model input size (default: 154 434)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flip-augment", action="store_true", default=True,
                        help="Average embeddings of original + horizontally flipped crop (default: on)")
    args = parser.parse_args()

    import random
    rng = random.Random(args.seed)

    review_root = Path(args.review)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    print(f"Loading model: {model_path}")
    model = torch.jit.load(str(model_path), map_location=args.device)
    model.eval()
    transform = build_transform(tuple(args.input_size))

    species_dirs = sorted(d for d in review_root.iterdir() if d.is_dir())

    centroids = []
    labels = []
    labels_keys = {}
    class_id = 0
    skipped = []

    for sp_dir in species_dirs:
        imgs = [p for p in sp_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if len(imgs) < args.min_crops:
            skipped.append(f"{sp_dir.name} ({len(imgs)} crops — below min {args.min_crops})")
            continue

        if len(imgs) > args.max_crops:
            imgs = rng.sample(imgs, args.max_crops)

        species_name = sp_dir.name.replace("_", " ")
        print(f"  {sp_dir.name}: {len(imgs)} crops → extracting embeddings…")

        embeddings = []
        for img_path in imgs:
            try:
                emb = extract_embedding(model, img_path, transform, args.device, args.flip_augment)
                embeddings.append(emb)
            except Exception as e:
                print(f"    ⚠  Skipping {img_path.name}: {e}")

        if not embeddings:
            skipped.append(f"{sp_dir.name} (all embeddings failed)")
            continue

        # Centroid = mean of normalized embeddings, then re-normalize
        centroid = F.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=0)
        centroids.append(centroid)
        labels.append(class_id)
        labels_keys[class_id] = {"name": species_name, "label": species_name}
        class_id += 1

    if not centroids:
        sys.exit("No species met the minimum crop threshold. Lower --min-crops.")

    gallery = {
        "centroids": torch.stack(centroids),   # (N_classes, embedding_dim)
        "labels": torch.tensor(labels),         # (N_classes,)
        "labels_keys": labels_keys,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gallery, str(output_path))

    print(f"\n✅ Gallery saved: {output_path}")
    print(f"   Species included ({len(centroids)}):")
    for cid, info in labels_keys.items():
        print(f"     {cid}: {info['name']}")
    if skipped:
        print(f"\n   Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"     {s}")

    print(f"\nTo use the gallery, run with:")
    print(f"  --cls-model {args.model} \\")
    print(f"  --cls-gallery {output_path} \\")
    print(f"  --cls-method natural_centroid")


if __name__ == "__main__":
    main()
