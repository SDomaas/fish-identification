# Species Galleries

Galleries are centroid files used by the ArcFace classifier to identify species.
Each gallery contains one embedding centroid per species, computed as the average
of all training crops for that species.

Built with `scripts/build_species_gallery.py` using flip augmentation (orientation invariant).

---

## gallery_full_side.pt

Crops where the **whole fish is visible from a side view** — full body, lateral profile.
Gives the cleanest, most representative centroids for species discrimination.

**Use this for production runs.**

---

## gallery_partial.pt

Crops with **partially visible fish** — fish partially out of frame, angled towards
the camera, or otherwise not showing a full lateral profile.

---

## local_gallery.pt

**Combination of full-side and partial crops.**

---

## Rebuilding a gallery

```bash
python scripts/build_species_gallery.py \
    --review /data/.../Species_library/review_full_side \
    --model classification_model/model.ts \
    --output classification_model/gallery_full_side.pt \
    --min-crops 10
```

Species library location:
`/data/P-Prosjekter/18200300_overvaking_av_laks_i_tanavassdraget/ARIS-data/2026/Karasjohka_video/training_crops/Species_library/`
