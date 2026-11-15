# Dataset acquisition notes

## FSOCO
https://fsoco.github.io/fsoco-dataset/download

- Ships **Supervisely** format by default (`ann/*.json`), not YOLO.
  `inventory.py` detects this and says so rather than reporting zero labels.
  Convert to YOLO before merging.
- 5 classes (blue / yellow / orange-small / orange-big / unknown cone) →
  all collapse to `cone`.
- Unpack to `data/raw/fsoco/`.

## AKS cone set (Roboflow Universe)
- Export as **YOLOv8/YOLO format**, not COCO.
- 2 classes: `cone` and a junk class literally named `1`.
  Task 2 decides whether `1` is junk or a real mislabel before it is dropped.
- Unpack to `data/raw/aks/`.
- Contains FSOCO-derived images — look for `gfr_*` and `bme_*` prefixes in the
  inventory's prefix histogram. Roboflow re-encodes, so filename matching is
  not enough; task 3 dedupes by perceptual hash and logs every collision.

## Run task 1 once both are unpacked

```bash
python3 scripts/inventory.py data/raw/fsoco --name fsoco
python3 scripts/inventory.py data/raw/aks   --name aks
```
