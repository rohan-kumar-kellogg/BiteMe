import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Inspect label distribution from data/dishes.csv")
    p.add_argument("--dishes_csv", default="data/dishes.csv")
    p.add_argument("--out_plot", default="reports/label_distribution.png")
    return p.parse_args()


def _resolve_dish_class_col(df: pd.DataFrame) -> str:
    if "dish_class" in df.columns:
        return "dish_class"
    if "dish_family" in df.columns:
        return "dish_family"
    if "dish_label" in df.columns:
        return "dish_label"
    raise ValueError("Expected `dish_class`, `dish_family`, or `dish_label` column in dishes CSV.")


def _print_counts(title: str, s: pd.Series, top_k: int = 20):
    print(f"\n{title} (top {top_k}):")
    vc = s.astype(str).value_counts()
    for label, count in vc.head(top_k).items():
        print(f"  {label}: {int(count)}")


def _print_threshold_stats(vc: pd.Series):
    n_lt5 = int((vc < 5).sum())
    n_lt10 = int((vc < 10).sum())
    n_lt20 = int((vc < 20).sum())
    print("\nClass sample thresholds:")
    print(f"  classes with <5 samples:  {n_lt5}")
    print(f"  classes with <10 samples: {n_lt10}")
    print(f"  classes with <20 samples: {n_lt20}")


def _save_histogram(vc: pd.Series, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            "\nCould not generate histogram plot because matplotlib is not installed. "
            "Install with: pip install matplotlib"
        )
        raise RuntimeError("matplotlib missing") from exc

    plt.figure(figsize=(10, 5))
    plt.hist(vc.values, bins=min(40, max(10, int(len(vc) ** 0.5))), color="#4e79a7", edgecolor="white")
    plt.title("Dish Class Sample Distribution")
    plt.xlabel("Samples per class")
    plt.ylabel("Number of classes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved histogram: {out_path}")


def main():
    args = parse_args()
    dishes_path = Path(args.dishes_csv)
    if not dishes_path.exists():
        raise FileNotFoundError(f"Missing file: {dishes_path}")

    df = pd.read_csv(dishes_path)
    class_col = _resolve_dish_class_col(df)
    if "cuisine" not in df.columns:
        raise ValueError("Expected `cuisine` column in dishes CSV.")

    n_images = len(df)
    n_classes = int(df[class_col].nunique(dropna=True))
    n_cuisines = int(df["cuisine"].nunique(dropna=True))
    print(f"Images: {n_images}")
    print(f"Unique {class_col}: {n_classes}")
    print(f"Unique cuisine: {n_cuisines}")

    _print_counts(class_col, df[class_col], top_k=20)
    _print_counts("cuisine", df["cuisine"], top_k=20)

    class_vc = df[class_col].astype(str).value_counts()
    _print_threshold_stats(class_vc)
    _save_histogram(class_vc, Path(args.out_plot))


if __name__ == "__main__":
    main()

