PYTHON ?= .venv/bin/python

.PHONY: train-tag-head eval-retrieval verify-pipeline data-quality

train-tag-head:
	$(PYTHON) utils/train_tag_head_from_manifest.py --manifest_csvs images/manifest.csv --out_ckpt data/models/clip_mlp_tag_head.pt --reports_dir reports --n_eval 500

eval-retrieval:
	$(PYTHON) utils/eval_retrieval.py --n_eval 500 --tag_head_ckpt data/models/clip_mlp_tag_head.pt --out_json reports/eval_retrieval.json

verify-pipeline:
	$(PYTHON) utils/verify_pipeline.py

data-quality:
	$(PYTHON) utils/data_quality_report.py --top_k 5 --out_csv reports/suspected_mislabeled.csv

