# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MEDRAP=0.0.0 uv sync --frozen --no-dev \
    && chmod +x /app/scripts/*.sh

RUN groupadd --system app \
    && useradd --system --gid app --create-home --shell /bin/bash app \
    && chown -R app:app /app

ENV PATH="/app/.venv/bin:${PATH}"

USER app

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
