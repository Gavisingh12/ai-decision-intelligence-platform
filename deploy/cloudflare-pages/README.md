# Cloudflare Pages Frontend

This folder helps you deploy the UI as a static site while keeping the FastAPI backend on a free Hugging Face Docker Space.

## Recommended free stack

- Frontend: Cloudflare Pages
- API + ML runtime: Hugging Face Docker Space
- Metadata database: Supabase Postgres
- Model and vector persistence: Hugging Face dataset repo mirroring

## Before you deploy

1. Deploy the backend first.
2. Make sure the backend URL works, for example:
   - `https://your-space-name.hf.space/api/v1/health`
3. Set backend CORS so the Cloudflare Pages domain is allowed:
   - `CORS_ORIGINS=https://your-project.pages.dev,http://localhost:8000,http://127.0.0.1:8000`

## Build the static bundle

The script below copies the current frontend into `deploy/cloudflare-pages/dist`, creates the `static/` folder that Cloudflare Pages expects, and writes `static/config.js` with your API base URL.

### PowerShell

```powershell
$env:PUBLIC_API_BASE_URL = "https://your-space-name.hf.space"
python deploy/cloudflare-pages/build_static_bundle.py
```

### Bash

```bash
PUBLIC_API_BASE_URL=https://your-space-name.hf.space python deploy/cloudflare-pages/build_static_bundle.py
```

## Deploy to Cloudflare Pages

### Fastest manual option

1. Run the build script locally.
2. Upload the contents of `deploy/cloudflare-pages/dist` to a new Cloudflare Pages project.

### Git-connected option

1. Connect the repository to Cloudflare Pages.
2. Use this build command:

```bash
python deploy/cloudflare-pages/build_static_bundle.py
```

3. Use this output directory:

```text
deploy/cloudflare-pages/dist
```

4. Add a build environment variable:

```text
PUBLIC_API_BASE_URL=https://your-space-name.hf.space
```

## What gets generated

- `dist/index.html`
- `dist/static/app.js`
- `dist/static/styles.css`
- `dist/static/config.js`
- `dist/_headers`
- `dist/_redirects`

## Notes

- Keep the backend on `FREE_MODE=true` for a lighter public demo.
- Free backends can sleep, so the UI already includes wake-up guidance.
- If you later move uploads or model files to Supabase Storage, the frontend bundle does not need to change.
