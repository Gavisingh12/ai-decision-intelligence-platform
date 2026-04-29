"""Build a Cloudflare Pages bundle for the static frontend."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"
DEPLOY_DIR = REPO_ROOT / "deploy" / "cloudflare-pages"
DIST_DIR = DEPLOY_DIR / "dist"
STATIC_DIR = DIST_DIR / "static"


def reset_dist_directory() -> None:
    """Remove the previous build output and recreate the directory tree."""

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)


def copy_file(source: Path, target: Path) -> None:
    """Copy one file while creating parent directories as needed."""

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_config(api_base_url: str) -> None:
    """Write the runtime config consumed by the static frontend."""

    payload = (
        "window.DECISION_ASSISTANT_CONFIG = window.DECISION_ASSISTANT_CONFIG || {\n"
        f"  apiBaseUrl: {api_base_url!r},\n"
        "};\n"
    )
    (STATIC_DIR / "config.js").write_text(payload, encoding="utf-8")


def main() -> None:
    """Build the static bundle used by Cloudflare Pages."""

    api_base_url = os.getenv("PUBLIC_API_BASE_URL", "").rstrip("/")
    reset_dist_directory()

    copy_file(FRONTEND_DIR / "index.html", DIST_DIR / "index.html")
    copy_file(FRONTEND_DIR / "app.js", STATIC_DIR / "app.js")
    copy_file(FRONTEND_DIR / "styles.css", STATIC_DIR / "styles.css")
    copy_file(DEPLOY_DIR / "_headers", DIST_DIR / "_headers")
    copy_file(DEPLOY_DIR / "_redirects", DIST_DIR / "_redirects")
    write_config(api_base_url)

    print(f"Built Cloudflare bundle in: {DIST_DIR}")
    if api_base_url:
        print(f"Configured API base URL: {api_base_url}")
    else:
        print("Warning: PUBLIC_API_BASE_URL is blank. The static app will call its own origin unless you update dist/static/config.js.")


if __name__ == "__main__":
    main()
