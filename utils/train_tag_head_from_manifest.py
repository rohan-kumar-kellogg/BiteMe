import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Train CLIP MLP tag head from one or more manifest files.")
    p.add_argument("--manifest_csvs", nargs="+", default=[])
    p.add_argument("--manifest_csv", default="", help="Single-manifest convenience alias.")
    p.add_argument("--tmp_labels_csv", default="data/cache/manifest_labels_for_tag_head.csv")
    p.add_argument("--out_ckpt", default="data/models/clip_mlp_tag_head.pt")
    p.add_argument("--reports_dir", default="reports")
    p.add_argument("--run_eval", action="store_true", default=True)
    p.add_argument("--n_eval", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    manifest_paths = list(args.manifest_csvs)
    if str(args.manifest_csv).strip():
        manifest_paths.append(str(args.manifest_csv))
    if not manifest_paths:
        manifest_paths = ["images/manifest.csv"]
    manifest_frames = []
    for path in manifest_paths:
        manifest_path = Path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        df = pd.read_csv(manifest_path)
        if "image_path" not in df.columns:
            raise ValueError(f"{manifest_path} must contain image_path")

        if "dish_label" in df.columns:
            dish_label = df["dish_label"].astype(str)
        elif "dish_class" in df.columns:
            dish_label = df["dish_class"].astype(str)
        elif "dish_family" in df.columns:
            dish_label = df["dish_family"].astype(str)
        else:
            raise ValueError(f"{manifest_path} needs one of: dish_label, dish_class, dish_family")

        src_col = (
            df["source"].astype(str)
            if "source" in df.columns
            else pd.Series([manifest_path.name] * len(df), index=df.index)
        )
        manifest_frames.append(
            pd.DataFrame(
                {
                    "image_path": df["image_path"].astype(str),
                    "dish_label": dish_label,
                    "cuisine": df["cuisine"].astype(str) if "cuisine" in df.columns else "",
                    "source": src_col,
                }
            )
        )

    tmp = pd.concat(manifest_frames, ignore_index=True)
    tmp = tmp.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    tmp = tmp[tmp["image_path"].map(lambda p: Path(str(p)).exists())].copy()
    Path(args.tmp_labels_csv).parent.mkdir(parents=True, exist_ok=True)
    tmp.to_csv(args.tmp_labels_csv, index=False)
    print(f"Prepared labels: {args.tmp_labels_csv} ({len(tmp)} rows)")
    print("Source distribution:")
    print(tmp["source"].value_counts().to_string())

    train_cmd = [
        sys.executable,
        "utils/train_clip_mlp_head.py",
        "--labels_csv",
        args.tmp_labels_csv,
        "--out_path",
        args.out_ckpt,
        "--reports_dir",
        args.reports_dir,
    ]
    subprocess.run(train_cmd, check=True)

    eval_json = Path(args.reports_dir) / "clip_mlp_eval.json"
    if eval_json.exists():
        with open(eval_json, "r") as f:
            rep = json.load(f)
        rep["cuisine_unknown_policy"] = "kept_as_class_if_meets_min_count"
        with open(eval_json, "w") as f:
            json.dump(rep, f, indent=2)
        print("- cuisine Unknown handling: kept as a class when it meets min_count.")
        print("\nTraining metrics:")
        for head in ["cuisine", "dish_family", "dish_class"]:
            if head in rep:
                print(
                    f"- {head}: top1={rep[head]['top1_accuracy']:.4f}, "
                    f"top3={rep[head]['top3_accuracy']:.4f}, n_eval={rep[head]['n_eval']}"
                )

    if args.run_eval:
        eval_cmd = [
            sys.executable,
            "utils/eval_retrieval.py",
            "--labels_csv",
            args.tmp_labels_csv,
            "--n_eval",
            str(args.n_eval),
            "--tag_head_ckpt",
            args.out_ckpt,
            "--out_json",
            str(Path(args.reports_dir) / "eval_retrieval.json"),
        ]
        subprocess.run(eval_cmd, check=True)
        with open(Path(args.reports_dir) / "eval_retrieval.json", "r") as f:
            ev = json.load(f)
        print("\nEval summary:")
        print(
            f"- retrieval-only: top1={ev['retrieval_only']['top1']:.4f}, "
            f"top3={ev['retrieval_only']['top3']:.4f}"
        )
        if ev.get("retrieval_mlp_rerank"):
            print(
                f"- retrieval+mlp rerank: top1={ev['retrieval_mlp_rerank']['top1']:.4f}, "
                f"top3={ev['retrieval_mlp_rerank']['top3']:.4f}"
            )
        if "source_breakdown" in ev:
            print("- source breakdown:")
            for k, v in ev["source_breakdown"].items():
                print(
                    f"  {k}: n={v['n_eval']} | retrieval_top1={v['retrieval_only_top1']:.4f} "
                    f"| mlp_top1={v['retrieval_mlp_top1']:.4f}"
                )


if __name__ == "__main__":
    main()

