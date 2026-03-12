# BiteMe FastAPI Backend

This adds a lightweight profile backend on top of the current model stack.

## What It Does

- Username-only create/load flow (no passwords)
- Upload food images over time
- Run existing prediction pipeline on each upload
- Persist profiles + upload events in SQLite
- Return user archetype plus multi-dimension inferred taste profile (preference analysis, not biological measurement)
- Compute compatibility with other users
- Return relative rankings within current app user base (e.g., more spice-seeking than most users)

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start API server:

```bash
uvicorn api.main:app --reload --port 8000
```

Docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Core Routes

- `POST /api/users/load_or_create`
  - body: `{"username":"rohan"}`
  - creates user if missing, otherwise loads existing user
- `POST /api/users/{username}/uploads`
  - multipart form with `image`
  - optional query: `confidence_threshold=0.86`
  - runs model prediction and updates profile
- `GET /api/users/{username}`
  - returns profile + recent uploads
- `GET /api/users/{username}/compatible?limit=5`
  - returns top compatible users by taste profile similarity with `why_you_match` explanations

## Persistence

- SQLite DB: `data/app_profiles.db`
- Uploaded images: `data/api_uploads/<username>/...`

## Notes

- Production runtime is FastAPI + Next.js.
- Legacy Streamlit tools were archived under `archive/legacy_streamlit/`.
- API uses baseline scoring mode by default for stability.

## Demo Data Seeding

Populate local demo users with rich, distinct taste profiles:

```bash
python scripts/seed_demo_users.py --reset_existing_demo
```

Optional custom DB path:

```bash
python scripts/seed_demo_users.py --db_path data/app_profiles.db --reset_existing_demo
```

Demo usernames:

- `demo_global_nomad`
- `demo_comfort_loyalist`
- `demo_dessert_first`
- `demo_protein_savory`
- `demo_spice_explorer`
- `demo_balanced_broad`
- `demo_fresh_greens`
- `demo_carb_crush`
