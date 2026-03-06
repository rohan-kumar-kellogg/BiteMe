import argparse
from pathlib import Path

import pandas as pd


def infer_prep_style(dish_class: str, course: str) -> str:
    d = str(dish_class).lower()
    c = str(course).lower()
    if any(k in d for k in ["tartare", "ceviche", "carpaccio", "sushi"]):
        return "raw-citrus"
    if any(k in d for k in ["grilled", "barbecue", "bbq", "kebab", "tandoori"]):
        return "grilled"
    if any(k in d for k in ["fried", "fries", "tempura", "samosa", "spring_roll"]):
        return "fried"
    if any(k in d for k in ["roast", "prime_rib", "chops"]):
        return "roasted"
    if any(k in d for k in ["soup", "curry", "stew", "pho", "ramen"]):
        return "stewed"
    if any(k in d for k in ["pasta", "risotto", "lasagna", "macaroni"]):
        return "simmered"
    if "dessert" in c:
        return "baked"
    return "mixed"


def normalize_protein(x: str) -> str:
    x = str(x).strip().lower()
    mp = {
        "meat": "meat",
        "seafood": "seafood",
        "plant": "plant",
        "mixed": "mixed",
    }
    return mp.get(x, "mixed")


def main():
    p = argparse.ArgumentParser(description="Bootstrap labels.csv from images/manifest.csv")
    p.add_argument("--manifest", default="images/manifest.csv")
    p.add_argument("--out", default="data/labels.csv")
    args = p.parse_args()

    mpath = Path(args.manifest)
    if not mpath.exists():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    df = pd.read_csv(mpath)
    required = {"image_path"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Manifest must include columns: {sorted(required)}")

    out = pd.DataFrame()
    out["image_path"] = df["image_path"].astype(str)
    out["dish_class"] = df["dish_class"].astype(str) if "dish_class" in df.columns else ""
    out["dish_family"] = df["dish_family"].astype(str) if "dish_family" in df.columns else ""
    if "dish_name" in df.columns:
        out["dish_name"] = df["dish_name"].astype(str)
    elif "dish_class" in df.columns:
        out["dish_name"] = df["dish_class"].astype(str).str.replace("_", " ", regex=False)
    else:
        out["dish_name"] = ""
    out["cuisine"] = df["cuisine"].astype(str).replace({"Unknown": "unknown"}) if "cuisine" in df.columns else ""
    out["protein_type"] = df["protein_type"].astype(str).map(normalize_protein) if "protein_type" in df.columns else ""
    if "course" in df.columns:
        out["course"] = df["course"].astype(str)
    else:
        out["course"] = ""
    out["prep_style"] = [
        infer_prep_style(dc, co) for dc, co in zip(out["dish_class"].astype(str), out["course"].astype(str))
    ]
    out["notes"] = "bootstrapped_from_manifest"

    opath = Path(args.out)
    opath.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(opath, index=False)
    print(f"Saved {len(out)} rows to {opath}")


if __name__ == "__main__":
    main()

