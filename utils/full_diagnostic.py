import importlib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.retrieval import predict_dish
from models.tag_head import CLIPTagPredictor
from models.vision import VisionEncoder
from utils.eval_on_personal_photos import run_personal_eval
from utils.eval_rerank import run_eval as run_rerank_eval
from utils.path_utils import normalize_path


def _version(name: str) -> str:
    try:
        m = importlib.import_module(name)
        return str(getattr(m, "__version__", "unknown"))
    except Exception:
        return "not_installed"


def _norm_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    checks: list[dict] = []
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    _print_header("1) Environment & Device")
    selected_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"mps_available: {torch.backends.mps.is_available()}")
    print(f"selected_device: {selected_device}")
    print("package_versions:")
    print(f"  numpy: {_version('numpy')}")
    print(f"  pandas: {_version('pandas')}")
    print(f"  torch: {_version('torch')}")
    print(f"  torchvision: {_version('torchvision')}")
    print(f"  transformers: {_version('transformers')}")
    print(f"  pillow: {_version('PIL')}")
    print(f"  hdbscan: {_version('hdbscan')}")
    checks.append({"name": "environment_printed", "ok": True, "detail": selected_device})

    _print_header("2) Artifact Existence & Schema")
    required_files = [
        Path("images/manifest.csv"),
        Path("data/dishes.csv"),
        Path("data/dish_vectors.npy"),
    ]
    optional_files = [
        Path("data/user_embeddings.npy"),
        Path("data/restaurant_embeddings.npy"),
        Path("data/models/clip_mlp_tag_head.pt"),
    ]
    req_missing = [str(p) for p in required_files if not p.exists()]
    if req_missing:
        print(f"Missing required artifacts: {req_missing}")
        checks.append({"name": "required_artifacts", "ok": False, "detail": ", ".join(req_missing)})
        # Without required artifacts, remaining checks cannot run reliably.
        _print_header("Summary")
        print("FAIL: required artifacts missing")
        sys.exit(1)
    print("Required artifacts: OK")
    for p in optional_files:
        print(f"Optional artifact {p}: {'present' if p.exists() else 'missing'}")
    checks.append({"name": "required_artifacts", "ok": True, "detail": "present"})

    manifest = pd.read_csv("images/manifest.csv")
    dishes = pd.read_csv("data/dishes.csv")
    need_cols = {"dish_label", "cuisine", "course", "protein_type", "image_path"}
    missing_cols = sorted(list(need_cols - set(dishes.columns)))
    if "dish_id" not in dishes.columns:
        dishes = dishes.reset_index(drop=False).rename(columns={"index": "dish_id"})
        print("dish_id column missing; inferred from index.")
    if missing_cols:
        print(f"Missing required dishes.csv columns: {missing_cols}")
        checks.append({"name": "dishes_schema", "ok": False, "detail": ",".join(missing_cols)})
    else:
        print("dishes.csv schema: OK")
        checks.append({"name": "dishes_schema", "ok": True, "detail": "ok"})

    vecs = np.load("data/dish_vectors.npy")
    expected_n = 6000
    counts_ok = len(manifest) == expected_n and len(dishes) == expected_n and (vecs.ndim == 2 and vecs.shape[0] == expected_n)
    checks.append(
        {
            "name": "dataset_size_6000",
            "ok": counts_ok,
            "detail": f"manifest={len(manifest)}, dishes={len(dishes)}, vec_rows={vecs.shape[0] if vecs.ndim==2 else -1}",
        }
    )
    if "source" in manifest.columns:
        src_counts = manifest["source"].astype(str).value_counts().to_dict()
        print(f"manifest source counts: {src_counts}")
        src_ok = int(src_counts.get("food101", 0)) == 3000 and int(src_counts.get("uecfood256", 0)) == 3000
        checks.append({"name": "source_counts_3000_each", "ok": src_ok, "detail": str(src_counts)})
    else:
        checks.append({"name": "source_counts_3000_each", "ok": False, "detail": "manifest missing source column"})
    if vecs.ndim != 2 or vecs.shape[0] != len(dishes):
        checks.append(
            {
                "name": "dish_vectors_shape",
                "ok": False,
                "detail": f"shape={vecs.shape}, dishes={len(dishes)}",
            }
        )
        print(f"dish_vectors mismatch: shape={vecs.shape} vs dishes={len(dishes)}")
    else:
        print(f"dish_vectors shape: {vecs.shape}")
        checks.append({"name": "dish_vectors_shape", "ok": True, "detail": str(vecs.shape)})
    is_float32 = vecs.dtype == np.float32
    checks.append({"name": "dish_vectors_dtype_float32", "ok": bool(is_float32), "detail": str(vecs.dtype)})
    print(f"dish_vectors dtype: {vecs.dtype}")
    norms = np.linalg.norm(vecs.astype(np.float32, copy=False), axis=1)
    print(f"L2 norms mean/min/max: {float(norms.mean()):.6f} / {float(norms.min()):.6f} / {float(norms.max()):.6f}")

    streamlit_text = Path("archive/legacy_streamlit/streamlit_app.py").read_text(encoding="utf-8")
    uses_bad_key = 'best.get("dish_class"' in streamlit_text
    checks.append(
        {
            "name": "streamlit_uses_dish_label",
            "ok": not uses_bad_key,
            "detail": "found dish_class key usage" if uses_bad_key else "dish_label usage",
        }
    )
    print(f"streamlit key usage check: {'OK' if not uses_bad_key else 'FAIL'}")

    _print_header("3) Preprocessing Consistency")
    encoder = VisionEncoder(device=selected_device)
    vecs_norm = _norm_rows(vecs.astype(np.float32, copy=False))
    existing = dishes[dishes["image_path"].map(lambda p: Path(str(p)).exists())].head(3)
    prep_fail = False
    for r in existing.itertuples(index=False):
        i = int(r.dish_id)
        emb_q = encoder.encode_image(str(r.image_path))
        cos = _cos(emb_q, vecs_norm[i])
        print(f"{r.image_path} -> cosine={cos:.6f}")
        if cos < 0.99:
            prep_fail = True
    checks.append({"name": "preprocessing_consistency_3_images", "ok": not prep_fail, "detail": "threshold 0.99"})

    _print_header("4) Retrieval Sanity")
    sample5 = dishes[dishes["image_path"].map(lambda p: Path(str(p)).exists())].head(5)
    exact_fail = False
    tagger_for_sanity = None
    tag_ckpt = Path("data/models/clip_mlp_tag_head.pt")
    if tag_ckpt.exists():
        try:
            tagger_for_sanity = CLIPTagPredictor(str(tag_ckpt))
        except Exception:
            tagger_for_sanity = None
    for r in sample5.itertuples(index=False):
        top_retrieval = predict_dish(
            str(r.image_path),
            dishes,
            vecs_norm,
            encoder=encoder,
            top_k=50,
            top_n=1,
            use_rerank=False,
            debug=True,
        )
        top_rerank = predict_dish(
            str(r.image_path),
            dishes,
            vecs_norm,
            encoder=encoder,
            tag_predictor=tagger_for_sanity,
            top_k=50,
            top_n=1,
            use_rerank=bool(tagger_for_sanity is not None),
            debug=True,
        )
        print(f"\nQuery: {r.image_path}")
        a = top_retrieval[0] if top_retrieval else {}
        b = top_rerank[0] if top_rerank else {}
        print(
            f"  retrieval-only: label={a.get('dish_label','')} "
            f"| sim={float(a.get('cosine_similarity', np.nan)):.4f} "
            f"| final={float(a.get('final_score', np.nan)):.4f} "
            f"| dish_agreement={a.get('dish_agreement', np.nan)} "
            f"| cuisine_agreement={a.get('cuisine_agreement', np.nan)}"
        )
        print(
            f"  rerank:        label={b.get('dish_label','')} "
            f"| sim={float(b.get('cosine_similarity', np.nan)):.4f} "
            f"| final={float(b.get('final_score', np.nan)):.4f} "
            f"| dish_agreement={b.get('dish_agreement', np.nan)} "
            f"| cuisine_agreement={b.get('cuisine_agreement', np.nan)}"
        )
        if not top_retrieval or float(top_retrieval[0].get("cosine_similarity", 0.0)) < 0.99:
            exact_fail = True
    checks.append({"name": "retrieval_exact_image_sim", "ok": not exact_fail, "detail": "top1 sim >= 0.99"})

    # Unit-ish self-exclusion check across random queries.
    candidates = dishes[dishes["image_path"].map(lambda p: Path(str(p)).exists())]
    n_self_test = min(10, len(candidates))
    self_exclusion_ok = True
    if n_self_test > 0:
        q10 = candidates.sample(n=n_self_test, random_state=42).reset_index(drop=True)
        for r in q10.itertuples(index=False):
            qn = normalize_path(str(r.image_path))
            top = predict_dish(
                str(r.image_path),
                dishes,
                vecs_norm,
                encoder=encoder,
                top_k=30,
                top_n=1,
                use_rerank=False,
                exclude_image_paths={qn},
            )
            if top and normalize_path(str(top[0].get("image_path", ""))) == qn:
                self_exclusion_ok = False
                print(f"Self-exclusion violation: query={r.image_path} returned itself as top1.")
                break
    checks.append({"name": "self_exclusion_top1_10_queries", "ok": self_exclusion_ok, "detail": "top1 != query"})

    _print_header("5) Tag Head / Rerank Path")
    tag_ok = True
    if tag_ckpt.exists():
        try:
            tagger = CLIPTagPredictor(str(tag_ckpt))
            mini = sample5.head(2)
            for r in mini.itertuples(index=False):
                tags = tagger.predict_tags(str(r.image_path), top_k=3)
                print(f"predict_tags({Path(str(r.image_path)).name}) keys={list(tags.keys())}")
            one = sample5.head(1).iloc[0]
            top_rr = predict_dish(
                str(one.image_path),
                dishes,
                vecs_norm,
                encoder=encoder,
                tag_predictor=tagger,
                top_k=50,
                top_n=3,
                use_rerank=True,
                debug=True,
            )
            fields_present = bool(top_rr and ("dish_agreement" in top_rr[0]) and ("final_score" in top_rr[0]))
            print(f"rerank fields present: {fields_present}")
            tag_ok = tag_ok and fields_present
        except Exception as e:
            print(f"Tag head check failed: {e}")
            tag_ok = False
    else:
        print("Tag head missing; using retrieval-only")
    checks.append({"name": "tag_head_rerank_path", "ok": tag_ok, "detail": "optional"})

    _print_header("6) Eval Harness Smoke Tests")
    eval_ok = True
    rr_path = reports_dir / "diagnostic_rerank_eval.json"
    pp_json = reports_dir / "diagnostic_personal_eval.json"
    pp_csv = reports_dir / "diagnostic_personal_failures.csv"
    try:
        run_rerank_eval(
            labels_csv="data/labels.csv",
            data_dir="data",
            n_eval=50,
            seed=42,
            tag_head_ckpt=str(tag_ckpt),
            out_json=str(rr_path),
            alpha_values="0.15",
        )
        print(f"created: {rr_path}")
    except Exception as e:
        print(f"rerank smoke eval failed: {e}")
        eval_ok = False
    try:
        run_personal_eval(
            manifest_csv="data/personal_manifest.csv",
            data_dir="data",
            tag_head_ckpt=str(tag_ckpt),
            n_eval=0,
            out_json=str(pp_json),
            out_failures=str(pp_csv),
        )
        print(f"created: {pp_json}")
        print(f"created: {pp_csv}")
    except Exception as e:
        print(f"personal eval smoke failed: {e}")
        eval_ok = False
    outputs_ok = rr_path.exists() and pp_json.exists() and pp_csv.exists()
    checks.append({"name": "eval_smoke_outputs", "ok": bool(eval_ok and outputs_ok), "detail": "reports/diagnostic_*"})

    _print_header("7) Summary")
    failed = [c for c in checks if not c["ok"]]
    print("Check Results:")
    for c in checks:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"- {mark:4} | {c['name']:<36} | {c['detail']}")

    print("\nTop 5 likely causes if accuracy is still poor:")
    likely = [
        "Dataset label noise or inconsistent class naming across manifests.",
        "No tag-head checkpoint loaded, so retrieval-only mode dominates.",
        "Cross-dataset label-space mismatch (aliases not unified yet).",
        "Insufficient hard negatives for visually similar plated proteins.",
        "Domain shift between training data composition and target real photos.",
    ]
    for i, item in enumerate(likely, start=1):
        print(f"{i}. {item}")

    if failed:
        print(f"\nOVERALL: FAIL ({len(failed)} checks failed)")
        sys.exit(1)
    print("\nOVERALL: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()

