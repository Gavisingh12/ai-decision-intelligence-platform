# AI Decision Intelligence Platform

Production-style FastAPI application for multi-modal analytics, combining structured time-series forecasting, label-based classification, document-grounded retrieval, SHAP explainability, and an AI decision assistant in one deployable web service.

## Recommended free architecture

This project now supports two free-friendly deployment shapes:

### Option A: simplest live setup

- Host the full app on a free Hugging Face Docker Space
- Keep SQLite for metadata
- Persist runtime files to a Hugging Face dataset repo with `HF_TOKEN` + `HF_PERSIST_REPO_ID`

This is the easiest way to get a public demo online with one service.

### Option B: cleaner public setup

- Host the static frontend on Cloudflare Pages
- Host the FastAPI backend on a free Hugging Face Docker Space
- Use Supabase Postgres for the metadata database
- Keep Hugging Face dataset mirroring for model and vector-store persistence

This split gives you a faster UI, a cleaner public URL, and a better story for free hosting without changing the app logic.

## What it does

- Uploads CSV datasets and stores structured records in SQLite by default, with PostgreSQL still supported
- Uploads PDF or text reports, chunks them, embeds them, and indexes them in FAISS
- Trains XGBoost forecasting models with lag, rolling, and calendar features
- Trains XGBoost classification models for recommendation or label prediction datasets
- Produces future predictions plus holdout evaluation metrics
- Generates SHAP global and local explainability artifacts
- Answers grounded questions through both a RAG endpoint and a decision-assistant endpoint
- Ships with a lightweight frontend served by FastAPI

## Architecture

```text
backend/
  api/routes/          REST endpoints
  core/                settings, logging, security
  db/                  SQLAlchemy models and session management
  explainability/      SHAP artifact generation
  rag/                 embeddings, FAISS store, retrieval pipeline
  schemas/             request/response models
  services/            ingestion, forecasting, assistant, storage, LLM
frontend/              web UI (HTML/CSS/JS)
data/
  sample_yield_timeseries.csv
  sample_report.txt
notebooks/
  README.md
Dockerfile
docker-compose.yml
render.yaml
requirements.txt
deploy/
```

## Core modules

- `backend/services/ingestion.py`: CSV parsing, dataset profiling, PDF/text extraction, document chunking
- `backend/services/forecasting.py`: XGBoost training, recursive forecasting, artifact persistence
- `backend/services/classification.py`: XGBoost classification training, feature preprocessing, confusion matrix and importance plots
- `backend/explainability/shap_service.py`: SHAP global importance and local explanation plots
- `backend/rag/pipeline.py`: retrieval and grounded answer generation
- `backend/services/assistant.py`: combines forecast signals, SHAP insights, and retrieved context

## Local setup

### Option 1: Docker Compose with PostgreSQL

1. Copy `.env.example` to `.env`
2. Add `OPENAI_API_KEY` or `GEMINI_API_KEY` if you want live LLM answers
3. Start the stack:

```bash
docker compose up --build
```

4. Open:

- App UI: [http://localhost:8000](http://localhost:8000)
- Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Local Python environment

1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. For quick local smoke tests, the app falls back to SQLite if `DATABASE_URL` is not provided. For the recommended split deployment path, point `DATABASE_URL` to Supabase Postgres or another managed Postgres instance.
4. Run the API:

```bash
uvicorn backend.main:app --reload
```

## Sample workflow

1. Register a user from the UI or `/api/v1/auth/register`
2. Upload `data/sample_yield_timeseries.csv`
3. Upload `data/sample_report.txt` or a PDF report
4. Train a forecast using:
   - `target_column=yield`
   - `time_column=date`
   - `horizon=14`
5. Ask:
   - `Why did yield drop?`
   - `What factors matter most?`
   - `What should be done next?`

For label-style datasets such as crop recommendation:

1. Upload a CSV with `target_column=label`
2. Leave `time_column` blank
3. Train `/api/v1/classification/train`
4. Use the Classification view to submit feature values and predict the best label

## Environment variables

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection string |
| `SECRET_KEY` | JWT signing key |
| `FREE_MODE` | Enables lighter defaults for cold starts and limited compute |
| `PUBLIC_API_BASE_URL` | API origin injected into `/static/config.js` for split frontend deployments |
| `REQUEST_LIMIT_PER_MINUTE` | Lightweight per-route request cap for public demos |
| `MAX_DOCUMENT_CHUNKS` | Maximum indexed text chunks per uploaded report |
| `SHAP_REFERENCE_SAMPLE_SIZE` | Sample size used for SHAP background data in free mode |
| `RECENT_TASK_LIMIT` | Number of recent activity items shown on the dashboard |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | OpenAI model name |
| `GEMINI_API_KEY` | Gemini API key |
| `GEMINI_MODEL` | Gemini model name |
| `DEFAULT_LLM_PROVIDER` | `openai` or `gemini` |
| `EMBEDDING_MODEL_NAME` | SentenceTransformer embedding model |
| `CORS_ORIGINS` | Comma-separated allowed origins |
| `HF_TOKEN` | Hugging Face token for optional free persistence |
| `HF_PERSIST_REPO_ID` | Dataset repo used to store runtime data snapshots |
| `HF_PERSIST_REPO_PRIVATE` | Whether the persistence dataset repo should be private |
| `FREE_DEPLOY_TARGET` | Free-host profile label, defaults to `huggingface-spaces` |

## Example API calls

### Register

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "demo@example.com",
    "password": "demo-pass-123",
    "full_name": "Demo User"
  }'
```

### Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo@example.com&password=demo-pass-123"
```

### Upload CSV

```bash
curl -X POST http://localhost:8000/api/v1/data/upload/csv \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@data/sample_yield_timeseries.csv" \
  -F "name=Regional Yield" \
  -F "target_column=yield" \
  -F "time_column=date"
```

### Upload document

```bash
curl -X POST http://localhost:8000/api/v1/data/upload/document \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@data/sample_report.txt" \
  -F "title=Weekly Agronomy Report"
```

### Train forecast

```bash
curl -X POST http://localhost:8000/api/v1/forecast/train \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "target_column": "yield",
    "time_column": "date",
    "horizon": 14,
    "lags": [1, 2, 3, 7, 14]
  }'
```

### Train classifier

```bash
curl -X POST http://localhost:8000/api/v1/classification/train \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "target_column": "label",
    "test_size": 0.2
  }'
```

### Predict label

```bash
curl -X POST http://localhost:8000/api/v1/classification/runs/1/predict \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "feature_values": {
      "N": 90,
      "P": 42,
      "K": 43,
      "temperature": 20.87,
      "humidity": 82.00,
      "ph": 6.50,
      "rainfall": 202.93
    }
  }'
```

### Run grounded RAG query

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What operational issues were reported in the field notes?",
    "top_k": 4
  }'
```

### Ask decision assistant

```bash
curl -X POST http://localhost:8000/api/v1/assistant/ask \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why did yield drop and what should be done next?",
    "dataset_id": 1,
    "forecast_run_id": 1,
    "top_k": 4
  }'
```

## Notes on behavior

- The forecasting flow uses recursive inference for future horizons.
- Classification runs accept categorical targets and reject obviously continuous numeric targets.
- Mixed datasets are handled more defensively: forecasting ignores unsupported categorical feature columns instead of crashing.
- Non-time covariates are forward-filled from the latest observation during future prediction.
- If no external LLM key is configured, the app returns a grounded local fallback answer rather than failing the request.
- If `sentence-transformers` is installed separately, the RAG layer uses semantic embeddings; otherwise it falls back to a lightweight hashing encoder and still stores vectors in FAISS.
- FAISS state is persisted under `data/vector_store/`.
- Forecast artifacts and SHAP plots are persisted under `data/models/`.
- Password hashing now uses `pbkdf2_sha256`, which avoids bcrypt's 72-byte limit and removes the registration error you hit.
- When `HF_TOKEN` and `HF_PERSIST_REPO_ID` are configured, runtime files are mirrored to a Hugging Face dataset repo so free Docker Space restarts do not wipe the working state.
- Uploads, training runs, and report indexing are now recorded as recent workspace activity so the UI can show progress and next steps clearly.
- The frontend uses guided dropdowns for CSV columns, separates Upload, Train, Results, and Ask AI into focused views, and avoids asking users to type technical column names when a safer choice exists.

## Testing

Run the smoke test suite:

```bash
pytest
```

## Deployment

### Free deployment path 1: one-service Hugging Face Space

1. Create a Docker Space on Hugging Face.
2. Push this project to that Space repository.
3. Set these Space secrets:
   - `SECRET_KEY`
   - `HF_TOKEN`
   - `HF_PERSIST_REPO_ID`
   - `OPENAI_API_KEY` or `GEMINI_API_KEY` if you want hosted LLM answers
4. Optional but recommended:
   - set `FREE_MODE=true`
   - keep `PUBLIC_API_BASE_URL` blank because the UI and API are served together
5. Copy [deploy/huggingface/SPACE_README.md](D:/Projects/AI%20Decision%20Intelligence%20Platform/deploy/huggingface/SPACE_README.md) into the Space repo root as `README.md` if you want Space card metadata.

This is the quickest fully free way to go live.

### GitHub-driven Hugging Face deployment

If you want GitHub to stay as the source of truth, this repository now includes:

- [D:\Projects\AI Decision Intelligence Platform\.github\workflows\sync-to-hf-space.yml](</D:/Projects/AI Decision Intelligence Platform/.github/workflows/sync-to-hf-space.yml>)

That workflow:

- runs on every push to `main`
- swaps in the Space-friendly README metadata before upload
- syncs the repo to a Hugging Face Docker Space with the official `huggingface/hub-sync` action

To use it, set:

- GitHub secret: `HF_TOKEN`
- GitHub variable: `HF_SPACE_REPO_ID` with a value like `your-hf-username/ai-decision-intelligence-platform`

### Free deployment path 2: Cloudflare Pages + Hugging Face Space + Supabase

1. Create a Supabase project and copy its Postgres connection string into `DATABASE_URL`.
2. Deploy this backend to a Hugging Face Docker Space with:
   - `DATABASE_URL=<supabase-postgres-url>`
   - `SECRET_KEY`
   - `HF_TOKEN`
   - `HF_PERSIST_REPO_ID`
   - `FREE_MODE=true`
3. Build the static frontend bundle:

```bash
PUBLIC_API_BASE_URL=https://your-space-name.hf.space python deploy/cloudflare-pages/build_static_bundle.py
```

On Windows PowerShell:

```powershell
$env:PUBLIC_API_BASE_URL = "https://your-space-name.hf.space"
python deploy/cloudflare-pages/build_static_bundle.py
```

4. Deploy `deploy/cloudflare-pages/dist` to Cloudflare Pages.
5. Use the generated `static/config.js` in that bundle to point the frontend at your Hugging Face backend.

This is the best free option when you want a cleaner public frontend and a lightweight API backend.

### Deployment assets included

- [Dockerfile](D:/Projects/AI%20Decision%20Intelligence%20Platform/Dockerfile): backend container for local Docker and Hugging Face Spaces
- [docker-compose.yml](D:/Projects/AI%20Decision%20Intelligence%20Platform/docker-compose.yml): local app + PostgreSQL stack
- [deploy/huggingface/SPACE_README.md](D:/Projects/AI%20Decision%20Intelligence%20Platform/deploy/huggingface/SPACE_README.md): Space metadata template
- [deploy/cloudflare-pages/README.md](D:/Projects/AI%20Decision%20Intelligence%20Platform/deploy/cloudflare-pages/README.md): static frontend deployment guide
- [deploy/cloudflare-pages/build_static_bundle.py](D:/Projects/AI%20Decision%20Intelligence%20Platform/deploy/cloudflare-pages/build_static_bundle.py): produces a Cloudflare-ready static bundle with API base configuration
- [deploy/cloudflare-pages/_headers](D:/Projects/AI%20Decision%20Intelligence%20Platform/deploy/cloudflare-pages/_headers): recommended response headers
- [deploy/cloudflare-pages/_redirects](D:/Projects/AI%20Decision%20Intelligence%20Platform/deploy/cloudflare-pages/_redirects): SPA fallback rule

## Suggested next hardening steps

- Move long-running forecast training and document indexing to a task queue such as Celery or Dramatiq
- Replace the in-memory request limiter and retrieval cache with Redis when you outgrow single-instance free hosting
- Add Alembic migrations for schema versioning
- Add Supabase Storage or S3-compatible object storage for large report files if you want to reduce local disk reliance further
- Add role-based permissions and audit logging for regulated environments
