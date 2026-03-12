from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from api.archetypes import CANONICAL_ARCHETYPE_NAMES, FALLBACK_ARCHETYPE, VALID_ARCHETYPE_NAMES, coerce_archetype_name
from api.profile_logic import infer_archetype
from api.storage import ProfileStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute all user archetypes using current canonical logic.")
    p.add_argument("--db_path", default="data/app_profiles.db")
    p.add_argument("--dry_run", action="store_true", help="Only report changes without writing.")
    return p.parse_args()


def migrate_all_users(db_path: str, dry_run: bool = False) -> dict[str, int]:
    store = ProfileStore(db_path)
    users = store.list_users()
    changed = 0
    total = len(users)
    before_counts = Counter()
    after_counts = Counter()
    invalid_before = 0

    for user in users:
        username = str(user["username"])
        before_name = str(user.get("archetype", "") or "")
        before_counts[before_name] += 1
        if before_name not in VALID_ARCHETYPE_NAMES:
            invalid_before += 1

        profile = dict(user.get("profile", {}))
        # Remove stale legacy helper fields if present.
        profile.pop("legacy_archetype", None)
        profile.pop("legacy_archetypes", None)
        if not profile.get("archetype_current"):
            profile["archetype_current"] = coerce_archetype_name(before_name, allow_system=False)
        profile["archetype_current"] = coerce_archetype_name(profile.get("archetype_current", ""), allow_system=False)

        archetype, desc, graphic, joke, observations = infer_archetype(profile)
        after_counts[archetype] += 1

        if (
            before_name != archetype
            or str(user.get("archetype_description", "")) != desc
            or str(user.get("archetype_graphic", "")) != graphic
            or str(user.get("joke", "")) != joke
            or str(user.get("observations", "")) != "\n".join(observations)
            or user.get("profile", {}) != profile
        ):
            changed += 1
            if not dry_run:
                store.upsert_user(
                    username=username,
                    email=str(user.get("email", "") or ""),
                    created_at=str(user.get("created_at", "")),
                    archetype=archetype,
                    archetype_description=desc,
                    archetype_graphic=graphic,
                    observations="\n".join(observations),
                    joke=joke,
                    profile=profile,
                )

    print(f"Users scanned: {total}")
    print(f"Users requiring archetype/profile rewrite: {changed}")
    print(f"Invalid archetype names before migration: {invalid_before}")
    print("Canonical archetypes:", ", ".join(CANONICAL_ARCHETYPE_NAMES))
    print("Fallback archetype:", FALLBACK_ARCHETYPE)
    print("\nDistinct archetypes before migration:")
    for name, n in sorted(before_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {name or '<empty>'}: {n}")
    print("\nDistinct archetypes after recompute:")
    for name, n in sorted(after_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {name}: {n}")
    if dry_run:
        print("\nDry run only. No rows were updated.")
    return {"total": total, "changed": changed, "invalid_before": invalid_before}


if __name__ == "__main__":
    args = parse_args()
    migrate_all_users(db_path=args.db_path, dry_run=bool(args.dry_run))
