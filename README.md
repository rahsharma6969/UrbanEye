# UrbanEye-ML (Starter)

Minimal ML layer for UrbanEye encroachment monitoring (Mumbai). Includes:
- STAC search (Planetary Computer)
- Sentinel-2/1 preprocessing and tiling
- Label rasterization
- Dataset + U-Net model
- Training + inference + vectorization
- ONNX export

## Quickstart
```bash
conda create -n urbeye python=3.11 -y
conda activate urbeye
pip install -r requirements.txt
# IMPORTANT: Install PyTorch per your platform (CPU/GPU)
# e.g. CPU-only: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 1) Configure AOIs and time windows
Edit `configs/config.yaml` (AOI paths, dates).

### 2) Search & sign imagery (STAC)
```bash
python -m src.stac.search_items --config configs/config.yaml --out outputs/stac_items.json
```

### 3) Generate chips (preprocess + tile)
```bash
python scripts/make_chips.py --config configs/config.yaml --items outputs/stac_items.json --out_index outputs/chips_index.parquet
```

### 4) Create labels in QGIS
Draw polygons of NEW built-up for t1 vs t0. Save to `data/labels/*.geojson`.
Rasterize:
```bash
python -m src.labeling.rasterize --config configs/config.yaml --index outputs/chips_index.parquet
```

### 5) Train model
```bash
python -m src.train.train --config configs/config.yaml --index outputs/chips_index.parquet
```

### 6) Inference (end-to-end on an AOI)
```bash
python -m src.infer.run --config configs/config.yaml --aoi data/aoi/mithi.geojson --t0 2024-01-01 2024-03-31 --t1 2025-06-01 2025-08-23 --out outputs/mithi_2025Q3.geojson
```

### 7) Export ONNX
```bash
python -m src.infer.export_onnx --weights outputs/model_best.pt --out models/urbeye_change.onnx
```

> NOTE: This is a starter. You’ll likely refine preprocessing, augmentations, and scoring in your app backend (PostGIS).
