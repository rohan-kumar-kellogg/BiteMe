from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from api.label_normalization import uec_category_map_report


def parse_args():
    p = argparse.ArgumentParser(description="Debug label normalization and confidence-weighted taste signals.")
    p.add_argument(
        "--labels",
        nargs="+",
        default=["242", "153", "apple_pie", "green curry"],
        help="Raw input labels to test.",
    )
    p.add_argument("--confidence", type=float, default=0.83)
    p.add_argument("--cuisine", default="")
    p.add_argument("--protein_type", default="")
    p.add_argument(
        "--validate_all_uec_ids",
        action="store_true",
        help="Validate that all 256 UEC class IDs are mapped from category.txt.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.validate_all_uec_ids:
        rep = uec_category_map_report(expected_max_id=256)
        print("UEC category map coverage:")
        print(f"- expected_count: {rep['expected_count']}")
        print(f"- mapped_count: {rep['mapped_count']}")
        print(f"- total_loaded_rows: {rep['total_loaded_rows']}")
        print(f"- is_complete_1_to_256: {rep['is_complete_1_to_256']}")
        if rep["missing_ids"]:
            print(f"- missing_ids: {rep['missing_ids']}")
        if rep["extra_ids"]:
            print(f"- extra_ids: {rep['extra_ids']}")
        print("=" * 40)

    try:
        from api.taste_profile import debug_applied_taste_traits
    except ModuleNotFoundError as exc:
        print(
            "Skipping trait-application debug output because required backend dependency is missing: "
            f"{exc}. Mapping coverage validation above is still valid."
        )
        return

    for raw_label in args.labels:
        row = debug_applied_taste_traits(
            raw_label,
            confidence=float(args.confidence),
            cuisine=args.cuisine,
            protein_type=args.protein_type,
        )
        print(f"Input label: {row['raw_label']}")
        print(f"Normalized label: {row['normalized_label']}")
        print(f"Confidence: {row['confidence']:.2f}")
        print("Applied traits:")
        traits = row.get("applied_traits", {})
        if not traits:
            print("- (no trait signal)")
        else:
            for k, v in sorted(traits.items(), key=lambda x: x[1], reverse=True):
                print(f"- {k}: {v:.4f}")
        print("-" * 40)


if __name__ == "__main__":
    main()
