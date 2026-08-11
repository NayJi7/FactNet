# The dashboard, as one container that answers both the API and the page.
#
# Model weights and collected data are deliberately NOT baked in. They are
# 1.9 GB, they change far less often than the code, and burying them in a layer
# would mean rebuilding and re-transferring all of it for a one-line fix. They
# are mounted read-only at run time instead; see compose.yaml.
#
# Built and run on arm64 as readily as on amd64: every wheel below publishes an
# aarch64 build, and torch on that platform is CPU-only by construction.

# ---------------------------------------------------------------- front end
# bun, because bun.lock is the lockfile this project actually keeps
FROM oven/bun:1-alpine AS web

WORKDIR /build
# the manifest and lockfile first, so a source edit does not reinstall anything
COPY web/package.json web/bun.lock ./
RUN bun install --frozen-lockfile

COPY web/ ./
RUN bun run build


# ---------------------------------------------------------------- runtime
FROM python:3.11-slim AS runtime

# uv resolves from the committed lockfile, so the deployed versions are the
# tested ones rather than whatever is current on the day of the build
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# dependencies as their own layer: they are the slow part and they change least
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY --from=web /build/dist ./web/dist

# an unprivileged owner for the code, and a writable home for the caches
# transformers insists on having even when it loads everything from disk
RUN useradd --system --uid 10001 --create-home --home-dir /home/factnet factnet \
 && chown -R factnet:factnet /app /home/factnet
USER factnet

ENV HF_HOME=/home/factnet/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    FACTNET_WEB_DIST=/app/web/dist \
    FACTNET_PORT=8347

EXPOSE 8347

# the container reports its own health, so the restart policy has something to
# act on when the engine is up but has no models to serve
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,json,os,sys; \
r=json.load(urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"FACTNET_PORT\"]}/api/health', timeout=8)); \
sys.exit(0 if r.get('ok') else 1)"

CMD ["sh", "-c", "exec uvicorn factnet.serve.api:app \
    --host 0.0.0.0 --port ${FACTNET_PORT} \
    --workers ${FACTNET_WORKERS:-1} \
    --proxy-headers --forwarded-allow-ips='*' \
    --access-log --log-level ${FACTNET_LOG_LEVEL:-info}"]
