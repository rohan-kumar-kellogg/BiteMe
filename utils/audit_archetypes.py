from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from api.archetypes import VALID_ARCHETYPE_NAMES
from api.storage import ProfileStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit stored user archetype values.")
    p.add_argument("--db_path", default="data/app_profiles.db")
    return p.parse_args()


def main(db_path: str) -> int:
    store = ProfileStore(db_path)
    users = store.list_users()
    counts = Counter(str(u.get("archetype", "") or "") for u in users)
    invalid = [(name, n) for name, n in counts.items() if name not in VALID_ARCHETYPE_NAMES]

    print(f"Users scanned: {len(users)}")
    print("Distinct stored archetypes:")
    for name, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {name or '<empty>'}: {n}")

    if invalid:
        print("\nInvalid archetypes found:")
        for name, n in sorted(invalid, key=lambda x: (-x[1], x[0])):
            print(f"  - {name or '<empty>'}: {n}")
        return 1

    print("\nAll stored archetypes are canonical.")
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args.db_path))
