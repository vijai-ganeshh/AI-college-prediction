# Version 2.0

## What Changed

- Rebuilt the Supabase/Postgres database from the JoSAA 2024 source and local NIRF data.
- Added a reusable bootstrap loader for schema + data setup.
- Switched the app to `Gemini 2.5 Flash`.
- Added `Tavily` web search for post-shortlist college context.
- Added the fine-tuned Hugging Face Gemma model for **summary drafting only**.
- Kept `Gemini` as the primary engine for:
  - query understanding
  - SQL generation
  - final answer polishing
- Improved rank-query handling with deterministic SQL for recommendation-style prompts.
- Added guards for `JEE Main` vs `JEE Advanced` confusion.
- Fixed closing-rank rendering issues in answers.
- Improved Streamlit chat management using native chat components.

## Current Flow

1. User asks a JoSAA counselling question.
2. App interprets the query with Gemini.
3. SQL is generated or routed through deterministic rank logic.
4. Supabase/Postgres returns matching JoSAA rows.
5. Tavily fetches supporting college web context.
6. Hugging Face Gemma drafts a summary.
7. Gemini polishes the final response shown in Streamlit.

## Database

- Main table: `josaa_btech_2024`
- Supporting table: `nirf_rankings_2025`
- Supporting table: `institute_mapping`
- Supporting views:
  - `josaa_2024`
  - `nirf_rankings_latest`
  - `josaa_nirf_combined`

## Environment Added

- `DB_HOST`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `DB_PORT`
- `DB_SSLMODE`
- `GEMINI_API_KEY`
- `TAVILY_API_KEY`
- `HF_TOKEN`
- `HF_MODEL_ID`
- `HF_MODEL_PATH`
- `USE_HF_SUMMARY`

## New Files / Key Updates

- [`scripts/bootstrap_supabase.py`](/teamspace/studios/this_studio/scripts/bootstrap_supabase.py)
- [`main.py`](/teamspace/studios/this_studio/main.py)
- [`requirements.txt`](/teamspace/studios/this_studio/requirements.txt)
- [`VERSION2.0.md`](/teamspace/studios/this_studio/VERSION2.0.md)

## Run

```bash
cd /teamspace/studios/this_studio
streamlit run main.py
```
