# Reviewer Guide

This document is a quick orientation for external code review.

## Production Runtime Path

- Backend: `api/main.py` (FastAPI)
- Frontend:
  - `frontend/app/layout.tsx`
  - `frontend/app/page.tsx`
  - `frontend/app/model-card/page.tsx`

## Core Production Code

- Backend core modules: `api/`, plus runtime model modules under `models/` used by `api/predictor.py`
- Frontend core modules: `frontend/app/`, `frontend/components/`, `frontend/lib/`

## Archived / Non-Runtime Artifacts

- `archive/legacy_streamlit/`
  - Legacy Streamlit apps kept for reference and optional older workflows.
- `archive/manual_tests/`
  - Manual or legacy test scripts moved out of core runtime directories.
- `archive/duplicates/`
  - Duplicated scripts retained for traceability, not part of primary workflow.

## Seed Script Canonical Choice

- Canonical demo seed script: `scripts/seed_demo_users.py`
- Legacy duplicate archived: `archive/duplicates/seed_demo_users.py`

## What Was Archived Instead of Deleted

- Legacy Streamlit apps
- Manual/legacy test scripts that are not part of production runtime
- Duplicate demo seed script

These were archived (not deleted) to preserve context and avoid risky removal during reviewer-readiness cleanup.
