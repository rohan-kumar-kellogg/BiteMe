# Improving Label Accuracy (Best ROI Path)

## Combined 8k Dataset (Food101 + UEC + Vireo)

Goal: build exactly 8000 labeled images with:
- 3000 from Food-101
- 3000 from UECFOOD256 (local)
- 2000 from VireoFood-172 (local)

No scraping is used.

### 1) Ensure local datasets are available

- Food-101 is downloaded automatically by torchvision in the build step.
- UEC and Vireo must already exist locally, for example:
  - `data/public_datasets/uecfood256`
  - `data/public_datasets/vireofood172`

Supported local layouts (UEC/Vireo):
- `labels.csv` with `image_path,dish_label` (optional `cuisine`)
- OR class folders where folder name is used as label

### 2) Build merged manifest (3k / 3k / 2k)

```bash
python utils/build_combined_manifest.py \
  --out_dir images_combined \
  --out_manifest images/manifest_merged.csv \
  --food101_root food101_data \
  --food101_split train \
  --food101_n 3000 \
  --uec_root data/public_datasets/uecfood256 \
  --uec_n 3000 \
  --vireo_root data/public_datasets/vireofood172 \
  --vireo_n 2000 \
  --seed 42 \
  --dedupe true \
  --copy_mode symlink
```

### 3) Generate embeddings from merged manifest

```bash
python utils/data_generator.py --manifest_csv images/manifest_merged.csv
```

### 4) Verify dataset growth + embedding consistency

```bash
python utils/verify_dataset_growth.py \
  --manifest_csv images/manifest_merged.csv \
  --dishes_csv data/dishes.csv \
  --dish_vectors data/dish_vectors.npy
```

### 5) Train tag head from merged manifest

```bash
python utils/train_tag_head_from_manifest.py \
  --manifest_csvs images/manifest_merged.csv \
  --out_ckpt data/models/clip_mlp_tag_head.pt \
  --reports_dir reports \
  --n_eval 500
```

### 6) Evaluate rerank on merged manifest

```bash
python utils/eval_rerank.py \
  --manifest_csv images/manifest_merged.csv \
  --data_dir data \
  --n_eval 500 \
  --tag_head_ckpt data/models/clip_mlp_tag_head.pt \
  --out_json reports/rerank_eval.json
```

### TODO: Cross-dataset label alias mapping

Current behavior:
- `source_dish_label` is preserved exactly from each dataset.
- `dish_label` is a normalized version of source label.

Optional future improvement:
- create `data/label_aliases.json` to map equivalent dishes across datasets
  (for example, unifying semantically identical labels with different names).

## Full project diagnostic (single command)

Run:

```bash
python utils/full_diagnostic.py
```

Outputs:
- `reports/diagnostic_rerank_eval.json`
- `reports/diagnostic_personal_eval.json`
- `reports/diagnostic_personal_failures.csv`

## QA TODO checklist

- [ ] Download/place local public datasets:
  - `data/public_datasets/uecfood256`
  - `data/public_datasets/vireofood172`
- [ ] Build merged manifest (`images/manifest_merged.csv`) with 3k/3k/2k split.
- [ ] Regenerate embeddings from merged manifest.
- [ ] Run `python utils/full_diagnostic.py` and confirm PASS.
- [ ] If needed, add `data/label_aliases.json` for cross-dataset dish mapping.

## End-to-end 6k rebuild (no personal photos, no scraping)

This path uses only:
- Food-101 via torchvision download
- local `public_datasets/uecfood256`

```bash
python utils/export_food101_manifest.py --n_total 3000 --seed 42
python utils/export_uecfood256_manifest.py --n_total 3000 --seed 42 --uec_root public_datasets/uecfood256
python utils/merge_manifests.py --inputs public_images/food101/manifest_food101.csv public_images/uecfood256/manifest_uecfood256.csv --out images/manifest.csv
python utils/data_generator.py --manifest_csv images/manifest.csv
python utils/train_tag_head_from_manifest.py --manifest_csv images/manifest.csv --out_ckpt data/models/clip_mlp_tag_head.pt --reports_dir reports --n_eval 500
python utils/eval_rerank.py --manifest_csv images/manifest.csv --tag_head_ckpt data/models/clip_mlp_tag_head.pt --n_eval 500 --top_k 50 --top_n 3
python utils/full_diagnostic.py
streamlit run archive/legacy_streamlit/streamlit_app.py
```

Expected strict counts:
- `food101=3000`
- `uecfood256=3000`
- `images/manifest.csv=6000`

## 1) Create a labeled CSV

Start from `data/labels_template.csv` and save as `data/labels.csv`.

Required:
- `image_path`

Recommended targets:
- `dish_name` (fine-grained)
- `cuisine`
- `protein` (e.g. seafood, lamb, chicken, veg)
- `prep_style` (e.g. grilled, roasted, raw-citrus, fried)

## 2) Train supervised heads on CLIP embeddings

```bash
python utils/train_clip_label_heads.py --labels_csv data/labels.csv --out_path data/models/label_heads.pkl
```

This trains lightweight classifiers using your labels and saves:
- `data/models/label_heads.pkl`

## 2b) Train frozen-CLIP MLP tag head (cuisine + dish_class)

```bash
python utils/train_clip_mlp_head.py \
  --labels_csv data/labels.csv \
  --cache_path data/cache/clip_embeddings_labels.npz \
  --out_path data/models/clip_mlp_tag_head.pt \
  --reports_dir reports
```

Outputs:
- `data/cache/clip_embeddings_labels.npz` (embedding cache)
- `data/models/clip_mlp_tag_head.pt` (MLP checkpoint + label maps)
- `reports/clip_mlp_eval.json`
- `reports/confusion_cuisine.csv`
- `reports/confusion_dish_class.csv`

## 3) Use in Streamlit automatically

`archive/legacy_streamlit/streamlit_app.py` will auto-load `data/models/label_heads.pkl` when present and use it to improve dish/cuisine predictions.

## 4) Deep model retraining (ResNet multi-task)

If CLIP heads still underperform, train a deeper model:

```bash
python utils/train_deep_multitask.py --labels_csv data/labels.csv --out_dir data/models/deep_multitask --epochs 8
```

On Apple Silicon (Mac), force GPU with:

```bash
python utils/train_deep_multitask.py --labels_csv data/labels.csv --out_dir data/models/deep_multitask --epochs 8 --device mps
```

On NVIDIA/CUDA systems:

```bash
python utils/train_deep_multitask.py --labels_csv data/labels.csv --out_dir data/models/deep_multitask --epochs 8 --device cuda
```

Artifacts:
- `data/models/deep_multitask/model.pt`
- `data/models/deep_multitask/meta.json`

## 5) Retrieval + rerank evaluation

```bash
python utils/eval_rerank.py \
  --labels_csv data/labels.csv \
  --data_dir data \
  --n_eval 200 \
  --alpha_values 0.05,0.10,0.15,0.20,0.30 \
  --tag_head_ckpt data/models/clip_mlp_tag_head.pt \
  --out_json reports/rerank_eval.json
```

CLI check for a single image:

```bash
python utils/test_predict_dish.py --image_path path/to/image.jpg --data_dir data --top_k 50 --top_n 3
```

## Real-photo accuracy upgrade workflow

### 1) Encode dataset with consistent preprocessing

```bash
python utils/data_generator.py
python utils/check_preprocessing_consistency.py --manifest_csv images/manifest.csv --dishes_csv data/dishes.csv --dish_vectors data/dish_vectors.npy
```

The consistency check should print:
- `OK: preprocessing consistency check passed.`

If not, it will warn:
- `Preprocessing mismatch: dataset embeddings not comparable to query embeddings.`

### 2) Train lightweight linear probes (dish + protein type)

```bash
python utils/train_probes_from_manifest.py \
  --manifest_csvs images/manifest.csv data/personal_manifest.csv \
  --out_path data/models/probes.pkl \
  --cache_path data/cache/probe_embeddings.npz \
  --reports_json reports/probes_eval.json
```

### 3) Run personal-photo ablations

```bash
python utils/eval_on_personal_photos.py \
  --manifest_csv data/personal_manifest.csv \
  --multi_crop true \
  --use_text_ensemble true \
  --use_protein_probe true \
  --out_json reports/personal_eval.json \
  --out_failures reports/personal_failures.csv
```

`reports/personal_eval.json` includes an ablation table:
- `retrieval-only`
- `+multi-crop`
- `+text ensemble`
- `+probe rerank`

### 4) Mine hard negatives from personal photos

```bash
python utils/mine_hard_negatives.py \
  --manifest_csv data/personal_manifest.csv \
  --use_text_ensemble \
  --use_protein_probe \
  --out_csv reports/hard_negatives.csv
```

This highlights high-similarity wrong predictions for targeted data cleanup.

### 5) Streamlit debug indicators

Run app:

```bash
streamlit run archive/legacy_streamlit/streamlit_app.py
```

Sidebar now shows:
- tag head loaded
- probe loaded
- encoder device
- rerank mode
- multi-crop / text-ensemble / protein-probe toggles

Each upload also shows a compact "Why this label" panel with:
- image-image similarity
- image-text similarity
- dish agreement
- protein agreement
- final score

## One-command tag-head training from manifest

Preferred (uses `images/manifest.csv` directly, trains, prints metrics, then runs eval):

```bash
python utils/train_tag_head_from_manifest.py --manifest_csvs images/manifest.csv --out_ckpt data/models/clip_mlp_tag_head.pt --reports_dir reports --n_eval 500
```
Add personal photos (20-50+) to `data/personal_manifest.csv` with:

```csv
image_path,dish_class,cuisine
```

To train with both base + personal labels, pass multiple manifests:

```bash
python utils/train_tag_head_from_manifest.py \
  --manifest_csvs images/manifest.csv data/personal_manifest.csv \
  --out_ckpt data/models/clip_mlp_tag_head.pt \
  --reports_dir reports \
  --n_eval 500
```

You can also merge public manifests first:

```bash
python utils/merge_manifests.py --inputs images/manifest.csv public_images/food101/manifest.csv --out images/manifest_merged.csv
python utils/train_tag_head_from_manifest.py --manifest_csvs images/manifest_merged.csv data/personal_manifest.csv --out_ckpt data/models/clip_mlp_tag_head.pt --reports_dir reports --n_eval 500
```

Or via Make:

```bash
make train-tag-head
```

This writes:
- `data/models/clip_mlp_tag_head.pt`
- `reports/clip_mlp_eval.json`
- `reports/confusion_*.csv`
- `reports/eval_retrieval.json`

## Label QC app

```bash
streamlit run archive/legacy_streamlit/streamlit_label_qc.py
```

Then:
1. Review `reports/suspected_mislabeled.csv`
2. Keep / Relabel / Remove
3. Export `images/manifest_clean.csv`
4. Regenerate data and re-evaluate:

```bash
python utils/data_generator.py
python utils/eval_retrieval.py --n_eval 500 --tag_head_ckpt data/models/clip_mlp_tag_head.pt --out_json reports/eval_retrieval.json
```

This model jointly predicts available columns from:
- `dish_name`
- `cuisine`
- `protein`
- `prep_style`

## 5) Data size guidance

- Prototype quality: ~500-1,500 labeled images
- Stronger production quality: 5k+ labeled images
- For deep training, `n_total=120` is too small.
- Prefer balanced sampling across dish classes.

## 6) Better taxonomy than only cuisine

Use a multi-attribute schema:
- `dish_name` (fine-grained)
- `cuisine` (coarse, optional if ambiguous)
- `protein` (seafood, lamb, chicken, veg, mixed)
- `prep_style` (grilled, roasted, stewed, fried, raw-citrus)
- optional: `dish_family` (salad, pasta, rice, sandwich, stew, dessert)

## Tips for better accuracy

- Add at least 5+ examples per class (more is better).
- Use clear class naming (e.g. `herb crusted lamb chops` instead of generic `lamb`).
- Keep labels consistent across similar dishes.
- Add "hard negatives" (visually similar but different dishes).
