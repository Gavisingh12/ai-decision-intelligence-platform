---
title: AI Decision Intelligence Platform
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
fullWidth: true
short_description: Multi-modal RAG, forecasting, and explainability workspace
---

# AI Decision Intelligence Platform

FastAPI-based decision intelligence workspace with:

- CSV + time-series ingestion
- PDF/text retrieval with FAISS
- XGBoost forecasting
- XGBoost classification for recommendation datasets
- SHAP explainability
- AI decision assistant

## Space role

This Space can run in either of these modes:

- full app: the Space serves both the UI and the API
- backend only: the Space serves the API while Cloudflare Pages serves the UI

## Recommended Space secrets

- `SECRET_KEY`
- `HF_TOKEN`
- `HF_PERSIST_REPO_ID`
- `DATABASE_URL` if you want Supabase Postgres instead of local SQLite
- `OPENAI_API_KEY` or `GEMINI_API_KEY` if you want hosted LLM answers

## Recommended Space variables

- `FREE_MODE=true`
- `FREE_DEPLOY_TARGET=huggingface-spaces`
- `PUBLIC_API_BASE_URL=` for the full app mode
- `CORS_ORIGINS=https://your-project.pages.dev,http://localhost:8000,http://127.0.0.1:8000` when using a separate Cloudflare Pages frontend

## Persistence

Free Hugging Face Spaces use ephemeral local disk, so this app can optionally mirror runtime data into a free Hugging Face dataset repository when `HF_TOKEN` and `HF_PERSIST_REPO_ID` are configured.
