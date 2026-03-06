from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.predictor import PredictionService
from api.profile_logic import (
    build_relative_rankings_for_user,
    compute_compatible_users,
    empty_profile,
    infer_archetype,
    update_profile_from_prediction,
)
from api.storage import ProfileStore, utc_now_iso
from api.taste_profile import generate_detailed_analysis, init_taste_profile


class UserInput(BaseModel):
    username: str = Field(..., min_length=2, max_length=64, description="Username only, no password.")


def _clean_username(x: str) -> str:
    out = "".join(ch for ch in str(x).strip() if ch.isalnum() or ch in {"_", "-", "."})
    return out.lower()


app = FastAPI(
    title="BiteMe Profile API",
    description=(
        "Lightweight consumer-facing backend for username-only profiles, food uploads, "
        "taste observations, archetypes, and compatibility matching."
    ),
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = ProfileStore("data/app_profiles.db")
predictor = PredictionService(data_dir="data", tag_head_ckpt="data/models/clip_mlp_tag_head.pt")
UPLOAD_ROOT = Path("data/api_uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_json(x):
    if isinstance(x, dict):
        return {k: _safe_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_safe_json(v) for v in x]
    if isinstance(x, tuple):
        return [_safe_json(v) for v in x]
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


def _user_payload(user: dict, compatible_users: list[dict], relative_rankings: list[dict]) -> dict:
    taste_profile = dict(user["profile"].get("taste_profile", {}))
    taste_profile["relative_rankings"] = relative_rankings
    return {
        "username": user["username"],
        "created_at": user["created_at"],
        "archetype": user["archetype"],
        "archetype_description": user["archetype_description"],
        "archetype_graphic": user["archetype_graphic"],
        "observations": user["observations"],
        "joke": user["joke"],
        "profile": {
            "upload_count": int(user["profile"].get("upload_count", 0)),
            "favorite_cuisines": user["profile"].get("favorite_cuisines", {}),
            "favorite_dishes": user["profile"].get("favorite_dishes", {}),
            "favorite_traits": user["profile"].get("favorite_traits", {}),
            "behavior_features": user["profile"].get("behavior_features", {}),
            "archetype_scores": user["profile"].get("archetype_scores", {}),
            "archetype_stability": user["profile"].get("archetype_stability", {}),
            "taste_profile": taste_profile,
            "relative_rankings": relative_rankings,
        },
        "compatible_users": compatible_users,
    }


def _load_or_create(username: str) -> tuple[dict, bool]:
    u = _clean_username(username)
    if len(u) < 2:
        raise HTTPException(status_code=400, detail="Username must be at least 2 valid characters.")
    existing = store.get_user(u)
    if existing is not None:
        # Backfill new taste-profile schema for older persisted rows.
        profile = existing.get("profile", {})
        init_taste_profile(profile)
        if "analysis" not in profile.get("taste_profile", {}):
            profile["taste_profile"]["analysis"] = generate_detailed_analysis(profile)
            store.upsert_user(
                username=existing["username"],
                created_at=existing["created_at"],
                archetype=existing["archetype"],
                archetype_description=existing["archetype_description"],
                archetype_graphic=existing["archetype_graphic"],
                observations=existing["observations"],
                joke=existing["joke"],
                profile=profile,
            )
            existing = store.get_user(u) or existing
        return existing, False
    profile = empty_profile()
    archetype, desc, graphic, joke, observations = infer_archetype(profile)
    created_at = utc_now_iso()
    store.upsert_user(
        username=u,
        created_at=created_at,
        archetype=archetype,
        archetype_description=desc,
        archetype_graphic=graphic,
        observations="\n".join(observations),
        joke=joke,
        profile=profile,
    )
    created = store.get_user(u)
    if created is None:
        raise HTTPException(status_code=500, detail="Failed to create user profile.")
    return created, True


@app.post("/api/users/load_or_create", summary="Create or load user by username")
def create_or_load_user(payload: UserInput):
    """
    Username-only auth flow:
    - if username exists -> load profile
    - if not -> auto-create profile
    """
    user, created = _load_or_create(payload.username)
    others = store.list_users(exclude_username=user["username"])
    compatible = compute_compatible_users(user, others, limit=5)
    rankings = build_relative_rankings_for_user(user, others)
    return _safe_json({"status": "ok", "created": created, "user": _user_payload(user, compatible, rankings)})


@app.post("/api/users/{username}/uploads", summary="Upload food image and update profile")
async def upload_food_image(
    username: str,
    image: UploadFile = File(...),
    confidence_threshold: float = Query(0.86, ge=0.5, le=0.99),
):
    """
    Upload one food image for a user.
    Runs the existing food prediction pipeline and updates persisted taste profile.
    """
    user, _ = _load_or_create(username)
    uname = user["username"]
    ext = Path(image.filename or "upload.jpg").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
        ext = ".jpg"

    user_dir = UPLOAD_ROOT / uname
    user_dir.mkdir(parents=True, exist_ok=True)
    out_path = user_dir / f"{utc_now_iso().replace(':', '')}_{os.urandom(3).hex()}{ext}"
    with open(out_path, "wb") as f:
        f.write(await image.read())

    pred = predictor.predict(str(out_path), confidence_threshold=float(confidence_threshold))
    updated_profile = update_profile_from_prediction(user["profile"], pred)
    archetype, desc, graphic, joke, observations = infer_archetype(updated_profile)

    store.upsert_user(
        username=uname,
        created_at=user["created_at"],
        archetype=archetype,
        archetype_description=desc,
        archetype_graphic=graphic,
        observations="\n".join(observations),
        joke=joke,
        profile=updated_profile,
    )
    upload_id = store.add_upload(username=uname, image_path=str(out_path), prediction=pred)
    latest = store.get_user(uname)
    if latest is None:
        raise HTTPException(status_code=500, detail="Failed to reload updated user.")
    others = store.list_users(exclude_username=uname)
    compatible = compute_compatible_users(latest, others, limit=5)
    rankings = build_relative_rankings_for_user(latest, others)
    return _safe_json({
        "status": "ok",
        "upload_id": upload_id,
        "prediction": pred,
        "user": _user_payload(latest, compatible, rankings),
    })


@app.get("/api/users/{username}", summary="Get user profile")
def get_user_profile(username: str):
    user, _ = _load_or_create(username)
    others = store.list_users(exclude_username=user["username"])
    compatible = compute_compatible_users(user, others, limit=5)
    rankings = build_relative_rankings_for_user(user, others)
    uploads = store.list_uploads(user["username"], limit=30)
    return _safe_json({"status": "ok", "user": _user_payload(user, compatible, rankings), "recent_uploads": uploads})


@app.get("/api/users/{username}/compatible", summary="Get top compatible users")
def get_compatible_users(username: str, limit: int = Query(5, ge=1, le=20)):
    user, _ = _load_or_create(username)
    others = store.list_users(exclude_username=user["username"])
    compatible = compute_compatible_users(user, others, limit=limit)
    return _safe_json({"status": "ok", "username": user["username"], "compatible_users": compatible})
