from __future__ import annotations

import os
import logging
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.archetypes import FALLBACK_ARCHETYPE, SYSTEM_ARCHETYPE, coerce_archetype_name, is_valid_archetype_name
from api.label_normalization import normalize_prediction_labels, normalize_profile_labels
from api.predictor import PredictionService
from api.profile_logic import (
    RECOMMENDATION_CLICK_SIGNAL_WEIGHT_DEFAULT,
    build_relative_rankings_for_user,
    compute_compatible_users,
    empty_profile,
    infer_archetype,
    update_profile_from_recommendation_click,
    update_profile_from_prediction,
)
from api.restaurant_recommendations import current_restaurant_source, get_compatible_restaurants
from api.storage import ProfileStore, utc_now_iso
from api.taste_profile import generate_detailed_analysis, init_taste_profile

LOGGER = logging.getLogger(__name__)

class UserInput(BaseModel):
    email: str = Field(..., min_length=3, max_length=254, description="Email for lightweight identity persistence.")
    username: str = Field(..., min_length=2, max_length=64, description="Username only, no password.")


class RecommendationFeedbackInput(BaseModel):
    dish_label: str = Field(..., min_length=2, max_length=128)
    cuisine: str = Field(default="", max_length=64)


class InviteInput(BaseModel):
    to_username: str = Field(..., min_length=2, max_length=64)
    to_email: str = Field(default="", max_length=254)
    restaurant_name: str = Field(default="", max_length=128)
    date: str = Field(default="", max_length=32)
    time: str = Field(default="", max_length=32)
    message: str = Field(default="", max_length=500)


def _clean_username(x: str) -> str:
    out = "".join(ch for ch in str(x).strip() if ch.isalnum() or ch in {"_", "-", "."})
    return out.lower()


def _clean_email(x: str) -> str:
    e = str(x or "").strip().lower()
    if not e or "@" not in e:
        raise HTTPException(status_code=400, detail="A valid email is required.")
    return e


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
    user_archetype = coerce_archetype_name(str(user.get("archetype", "") or ""), allow_system=True)
    taste_profile = dict(user["profile"].get("taste_profile", {}))
    taste_profile["relative_rankings"] = relative_rankings
    return {
        "username": user["username"],
        "email": str(user.get("email", "") or ""),
        "created_at": user["created_at"],
        "archetype": user_archetype,
        "primary_archetype": user["profile"].get("primary_archetype", user_archetype),
        "secondary_trait": user["profile"].get("secondary_trait", ""),
        "archetype_description": user["archetype_description"],
        "archetype_graphic": user["archetype_graphic"],
        "observations": user["observations"],
        "joke": user["joke"],
        "profile": {
            "upload_count": int(user["profile"].get("upload_count", 0)),
            "interaction_count": int(user["profile"].get("interaction_count", 0)),
            "favorite_cuisines": user["profile"].get("favorite_cuisines", {}),
            "favorite_dishes": user["profile"].get("favorite_dishes", {}),
            "favorite_traits": user["profile"].get("favorite_traits", {}),
            "recommendation_feedback": user["profile"].get("recommendation_feedback", []),
            "primary_archetype": user["profile"].get("primary_archetype", user_archetype),
            "secondary_trait": user["profile"].get("secondary_trait", ""),
            "behavior_features": user["profile"].get("behavior_features", {}),
            "archetype_scores": user["profile"].get("archetype_scores", {}),
            "archetype_stability": user["profile"].get("archetype_stability", {}),
            "taste_profile": taste_profile,
            "relative_rankings": relative_rankings,
        },
        "compatible_users": compatible_users,
    }


def _upload_debug_payload(pred: dict) -> dict:
    return {
        "predicted_label": str(pred.get("predicted_label", "") or ""),
        "abstained": bool(pred.get("abstained", False)),
        "rejected_as_not_food": bool(pred.get("rejected_as_not_food", False)),
        "food_gate_score": (
            float(pred.get("food_gate_score"))
            if pred.get("food_gate_score") is not None
            else None
        ),
        "confidence": (
            float(pred.get("predicted_score"))
            if pred.get("predicted_score") is not None
            else None
        ),
        "top1_top2_margin": (
            float(pred.get("top1_top2_margin"))
            if pred.get("top1_top2_margin") is not None
            else None
        ),
    }


def _normalize_recommendation_feedback_events(profile: dict) -> tuple[list[dict], bool]:
    events = list(profile.get("recommendation_feedback", []))
    changed = False
    out: list[dict] = []
    for e in events:
        row = dict(e) if isinstance(e, dict) else {}
        if str(row.get("event_type", "")) == "recommendation_click" and not str(row.get("event_id", "")).strip():
            row["event_id"] = str(uuid4())
            changed = True
        out.append(row)
    return out, changed


def _load_or_create(username: str, email: str | None = None) -> tuple[dict, bool]:
    u = _clean_username(username)
    clean_email = _clean_email(email) if email is not None else ""
    if len(u) < 2:
        raise HTTPException(status_code=400, detail="Username must be at least 2 valid characters.")
    existing = store.get_user(u)
    if existing is not None:
        # Backfill new taste-profile schema for older persisted rows.
        profile, labels_changed = normalize_profile_labels(existing.get("profile", {}))
        init_taste_profile(profile)
        normalized_feedback, feedback_ids_changed = _normalize_recommendation_feedback_events(profile)
        profile["recommendation_feedback"] = normalized_feedback
        click_events_count = sum(
            1 for e in normalized_feedback if str(e.get("event_type", "")) == "recommendation_click"
        )
        inferred_interactions = int(profile.get("upload_count", 0) or 0) + int(click_events_count)
        prev_interactions = int(profile.get("interaction_count", 0) or 0)
        profile["interaction_count"] = max(prev_interactions, inferred_interactions)
        if profile["interaction_count"] != prev_interactions:
            labels_changed = True
        prev_current = str(profile.get("archetype_current", "") or "")
        if not is_valid_archetype_name(prev_current, allow_system=True):
            profile["archetype_current"] = FALLBACK_ARCHETYPE
            labels_changed = True

        archetype_invalid = not is_valid_archetype_name(str(existing.get("archetype", "") or ""), allow_system=True)
        needs_analysis = "analysis" not in profile.get("taste_profile", {})
        history = profile.get("taste_profile", {}).get("history", [])
        needs_history_seed = bool(int(profile.get("upload_count", 0)) > 0 and not history)
        if needs_analysis:
            profile["taste_profile"]["analysis"] = generate_detailed_analysis(profile)
        if needs_history_seed:
            dims = profile.get("taste_profile", {}).get("dimensions", {})
            profile["taste_profile"]["history"] = [
                {
                    "timestamp": utc_now_iso(),
                    "upload_count": int(profile.get("upload_count", 0)),
                    "interaction_count": int(profile.get("interaction_count", 0) or profile.get("upload_count", 0)),
                    "dimensions": {
                        "sweet_leaning": float(dims.get("sweet_leaning", {}).get("score", 0.5)),
                        "spicy_leaning": float(dims.get("spicy_leaning", {}).get("score", 0.5)),
                        "richness_preference": float(dims.get("richness_preference", {}).get("score", 0.5)),
                        "freshness_preference": float(dims.get("freshness_preference", {}).get("score", 0.5)),
                        "dessert_affinity": float(dims.get("dessert_affinity", {}).get("score", 0.5)),
                    },
                }
            ]

        if labels_changed or archetype_invalid:
            archetype, desc, graphic, joke, observations = infer_archetype(profile)
            existing["archetype"] = archetype
            existing["archetype_description"] = desc
            existing["archetype_graphic"] = graphic
            existing["joke"] = joke
            existing["observations"] = "\n".join(observations)
        else:
            existing["archetype"] = coerce_archetype_name(existing.get("archetype", ""), allow_system=True)
            if not existing.get("archetype"):
                existing["archetype"] = SYSTEM_ARCHETYPE if int(profile.get("upload_count", 0)) <= 0 else FALLBACK_ARCHETYPE

        if clean_email and clean_email != str(existing.get("email", "")).strip().lower():
            existing["email"] = clean_email
            labels_changed = True

        if labels_changed or needs_analysis or needs_history_seed or feedback_ids_changed:
            store.upsert_user(
                username=existing["username"],
                email=str(existing.get("email", "") or clean_email),
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
        email=clean_email,
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


def _rebuild_profile_from_signals(username: str, feedback_events: list[dict] | None = None) -> dict:
    user = store.get_user(username)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    uploads = store.list_uploads(username, limit=100000)
    uploads_sorted = sorted(uploads, key=lambda r: int(r.get("id", 0)))
    profile = empty_profile()
    for row in uploads_sorted:
        pred = normalize_prediction_labels(row.get("prediction", {}))
        profile = update_profile_from_prediction(profile, pred)

    events = list(feedback_events if feedback_events is not None else user.get("profile", {}).get("recommendation_feedback", []))
    events, _ = _normalize_recommendation_feedback_events({"recommendation_feedback": events})
    events_sorted = sorted(events, key=lambda e: str(e.get("timestamp", "")))
    for e in events_sorted:
        if str(e.get("event_type", "")) != "recommendation_click":
            continue
        profile = update_profile_from_recommendation_click(
            profile,
            dish_label=str(e.get("dish_label", "")),
            cuisine=str(e.get("cuisine", "")),
            signal_weight=float(
                e.get("signal_weight", RECOMMENDATION_CLICK_SIGNAL_WEIGHT_DEFAULT)
                or RECOMMENDATION_CLICK_SIGNAL_WEIGHT_DEFAULT
            ),
            record_event=False,
            event_id=str(e.get("event_id", "")),
            timestamp=str(e.get("timestamp", "")),
        )
    profile["recommendation_feedback"] = events_sorted[-200:]
    return profile


@app.post("/api/users/load_or_create", summary="Create or load user by email + username")
def create_or_load_user(payload: UserInput):
    """
    Lightweight identity flow:
    - if username exists -> load profile
    - if not -> auto-create profile
    """
    user, created = _load_or_create(payload.username, email=payload.email)
    others = store.list_users(exclude_username=user["username"])
    compatible = compute_compatible_users(user, others, limit=5)
    rankings = build_relative_rankings_for_user(user, others)
    return _safe_json({"status": "ok", "created": created, "user": _user_payload(user, compatible, rankings)})


@app.post("/api/users/{username}/uploads", summary="Upload food image and update profile")
async def upload_food_image(
    username: str,
    image: UploadFile = File(...),
    confidence_threshold: float = Query(0.86, ge=0.5, le=0.99),
    debug_metadata: bool = Query(False, description="Include lightweight upload debug fields in response."),
):
    """
    Upload one food image for a user.
    Runs the existing food prediction pipeline and updates persisted taste profile.
    """
    LOGGER.info("upload endpoint entered: username=%s filename=%s", username, image.filename or "")
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
    pred = normalize_prediction_labels(pred)
    updated_profile = update_profile_from_prediction(user["profile"], pred)
    archetype, desc, graphic, joke, observations = infer_archetype(updated_profile)

    store.upsert_user(
        username=uname,
        email=str(user.get("email", "") or ""),
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
    response_payload = {
        "status": "ok",
        "upload_id": upload_id,
        "prediction": pred,
        "user": _user_payload(latest, compatible, rankings),
    }

    debug_payload = _upload_debug_payload(pred)
    # Optional response field for quick QA validation; ignored by normal UI.
    if debug_metadata:
        response_payload["upload_debug"] = debug_payload

    # Optional dev logging (disabled by default). Set BITEME_UPLOAD_DEBUG_LOGS=1 to enable.
    if os.getenv("BITEME_UPLOAD_DEBUG_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}:
        LOGGER.info("Upload debug metadata: %s", debug_payload)

    LOGGER.info(
        "upload endpoint returning: username=%s upload_id=%s predicted_label=%s abstained=%s",
        uname,
        upload_id,
        str(pred.get("predicted_label", "")),
        bool(pred.get("abstained", False)),
    )
    return _safe_json(response_payload)


@app.get("/api/users/{username}", summary="Get user profile")
def get_user_profile(username: str):
    LOGGER.info("profile endpoint entered: username=%s", username)
    user, _ = _load_or_create(username)
    others = store.list_users(exclude_username=user["username"])
    compatible = compute_compatible_users(user, others, limit=5)
    rankings = build_relative_rankings_for_user(user, others)
    uploads = store.list_uploads(user["username"], limit=30)
    for row in uploads:
        row["prediction"] = normalize_prediction_labels(row.get("prediction", {}))
    LOGGER.info(
        "profile endpoint returning: username=%s uploads=%s",
        user["username"],
        len(uploads),
    )
    return _safe_json({"status": "ok", "user": _user_payload(user, compatible, rankings), "recent_uploads": uploads})


@app.get("/api/users/{username}/compatible", summary="Get top compatible users")
def get_compatible_users(username: str, limit: int = Query(5, ge=1, le=20)):
    user, _ = _load_or_create(username)
    others = store.list_users(exclude_username=user["username"])
    compatible = compute_compatible_users(user, others, limit=limit)
    return _safe_json({"status": "ok", "username": user["username"], "compatible_users": compatible})


@app.get("/api/users/{username}/restaurants", summary="Get compatible restaurant recommendations")
def get_restaurants_for_user(
    username: str,
    zip_code: str = Query("", min_length=0, max_length=10),
    context: str = Query("", min_length=0, max_length=32),
    limit: int = Query(20, ge=1, le=50),
):
    user, _ = _load_or_create(username)
    restaurants = get_compatible_restaurants(user.get("profile", {}), zip_code=zip_code, limit=limit, context=context)
    source = current_restaurant_source()
    return _safe_json(
        {
            "status": "ok",
            "username": user["username"],
            "zip_code": zip_code,
            "context": context,
            "source": source,
            "fallback": "empty" if not restaurants else "none",
            "restaurants": restaurants,
        }
    )


@app.post("/api/users/{username}/recommendations/feedback", summary="Record recommendation click feedback")
def add_recommendation_feedback(username: str, payload: RecommendationFeedbackInput):
    user, _ = _load_or_create(username)
    uname = user["username"]
    updated_profile = update_profile_from_recommendation_click(
        user["profile"],
        dish_label=payload.dish_label,
        cuisine=payload.cuisine,
        event_id=str(uuid4()),
        timestamp=utc_now_iso(),
    )
    archetype, desc, graphic, joke, observations = infer_archetype(updated_profile)
    store.upsert_user(
        username=uname,
        email=str(user.get("email", "") or ""),
        created_at=user["created_at"],
        archetype=archetype,
        archetype_description=desc,
        archetype_graphic=graphic,
        observations="\n".join(observations),
        joke=joke,
        profile=updated_profile,
    )
    latest = store.get_user(uname)
    if latest is None:
        raise HTTPException(status_code=500, detail="Failed to reload updated user.")
    others = store.list_users(exclude_username=uname)
    compatible = compute_compatible_users(latest, others, limit=5)
    rankings = build_relative_rankings_for_user(latest, others)
    return _safe_json({"status": "ok", "user": _user_payload(latest, compatible, rankings)})


@app.delete("/api/users/{username}/recommendations/feedback/{event_id}", summary="Remove recommendation click feedback")
def remove_recommendation_feedback(username: str, event_id: str):
    user, _ = _load_or_create(username)
    uname = user["username"]
    events = list(user.get("profile", {}).get("recommendation_feedback", []))
    event_id_str = str(event_id)

    # Primary path: real persistent event_id.
    filtered = [e for e in events if str(e.get("event_id", "")) != event_id_str]
    found = len(filtered) != len(events)

    # Legacy fallback: some older rows had no event_id and frontend synthesized "<timestamp>_<index>".
    if not found:
        filtered = []
        for i, e in enumerate(events):
            legacy_id = f"{str(e.get('timestamp', '') or 'evt')}_{i}"
            if legacy_id != event_id_str:
                filtered.append(e)
            else:
                found = True

    if not found:
        raise HTTPException(status_code=404, detail="Recommendation click not found.")

    rebuilt_profile = _rebuild_profile_from_signals(uname, feedback_events=filtered)
    archetype, desc, graphic, joke, observations = infer_archetype(rebuilt_profile)
    store.upsert_user(
        username=uname,
        email=str(user.get("email", "") or ""),
        created_at=user["created_at"],
        archetype=archetype,
        archetype_description=desc,
        archetype_graphic=graphic,
        observations="\n".join(observations),
        joke=joke,
        profile=rebuilt_profile,
    )
    latest = store.get_user(uname)
    if latest is None:
        raise HTTPException(status_code=500, detail="Failed to reload updated user.")
    others = store.list_users(exclude_username=uname)
    compatible = compute_compatible_users(latest, others, limit=5)
    rankings = build_relative_rankings_for_user(latest, others)
    return _safe_json({"status": "ok", "user": _user_payload(latest, compatible, rankings)})


@app.delete("/api/users/{username}/uploads/{upload_id}", summary="Remove uploaded dish")
def remove_upload(username: str, upload_id: int):
    user, _ = _load_or_create(username)
    uname = user["username"]
    removed = store.delete_upload(username=uname, upload_id=int(upload_id))
    if not removed:
        raise HTTPException(status_code=404, detail="Upload not found.")

    rebuilt_profile = _rebuild_profile_from_signals(uname)
    archetype, desc, graphic, joke, observations = infer_archetype(rebuilt_profile)
    store.upsert_user(
        username=uname,
        email=str(user.get("email", "") or ""),
        created_at=user["created_at"],
        archetype=archetype,
        archetype_description=desc,
        archetype_graphic=graphic,
        observations="\n".join(observations),
        joke=joke,
        profile=rebuilt_profile,
    )
    latest = store.get_user(uname)
    if latest is None:
        raise HTTPException(status_code=500, detail="Failed to reload updated user.")
    others = store.list_users(exclude_username=uname)
    compatible = compute_compatible_users(latest, others, limit=5)
    rankings = build_relative_rankings_for_user(latest, others)
    return _safe_json({"status": "ok", "user": _user_payload(latest, compatible, rankings)})


@app.post("/api/users/{username}/invites", summary="Create a lightweight invite-to-eat record")
def create_invite(username: str, payload: InviteInput):
    user, _ = _load_or_create(username)
    from_username = user["username"]
    to_username = _clean_username(payload.to_username)
    if len(to_username) < 2:
        raise HTTPException(status_code=400, detail="Recipient username is required.")
    to_user = store.get_user(to_username)
    to_email = _clean_email(payload.to_email) if payload.to_email.strip() else str((to_user or {}).get("email", "") or "")
    invite_id = store.add_invite(
        from_username=from_username,
        to_username=to_username,
        to_email=to_email,
        restaurant_name=str(payload.restaurant_name or "").strip(),
        invite_date=str(payload.date or "").strip(),
        invite_time=str(payload.time or "").strip(),
        message=str(payload.message or "").strip(),
        status="pending",
    )
    return _safe_json(
        {
            "status": "ok",
            "invite": {
                "id": invite_id,
                "from_username": from_username,
                "to_username": to_username,
                "to_email": to_email,
                "restaurant_name": str(payload.restaurant_name or "").strip(),
                "date": str(payload.date or "").strip(),
                "time": str(payload.time or "").strip(),
                "message": str(payload.message or "").strip(),
                "status": "pending",
                "created_at": utc_now_iso(),
            },
        }
    )
