FROM python:3.12-slim

WORKDIR /app

# Local Kokoro TTS (torch + CUDA wheels + a 327MB model) is a DEV convenience so
# we don't spend ElevenLabs credits while iterating. Production TTS is ElevenLabs
# over HTTP and needs none of it — build prod (and the Celery worker) with
# INSTALL_LOCAL_TTS=false for a lean, torch-free image.
ARG INSTALL_LOCAL_TTS=true

# System dependencies:
#   nodejs / npm  — spawn the marcelmarais/spotify-mcp-server over MCP stdio
#                   (it is a Node process; the api container is its parent).
#   espeak-ng     — Kokoro's grapheme→phoneme fallback for out-of-vocabulary words.
# python:3.12-slim is Debian trixie, which ships Node 20 (>= the MCP SDK's 18).
RUN apt-get update && apt-get install -y --no-install-recommends \
        nodejs npm espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# The dependency set is heavy (torch + CUDA wheels + Kokoro) and saturates the
# network, so uv's 30s default HTTP timeout aborts on a slow wheel (orjson seen
# timing out mid-sync). Raise it; also give HuggingFace headroom for the 327MB
# Kokoro weights pulled in the warm-up step below.
ENV UV_HTTP_TIMEOUT=600
ENV HF_HUB_DOWNLOAD_TIMEOUT=120

COPY pyproject.toml uv.lock* ./

# Core deps, plus the local-tts extra (Kokoro + soundfile, which pulls torch)
# ONLY when INSTALL_LOCAL_TTS=true. Prod / the worker skip it → no torch/CUDA.
RUN if [ "$INSTALL_LOCAL_TTS" = "true" ]; then \
        uv sync --frozen --no-dev --extra local-tts; \
    else \
        uv sync --frozen --no-dev; \
    fi

# Pre-download Kokoro's model + default voice INTO the image (one synth warms
# the 327MB weights + af_heart voice into /root/.cache/huggingface) so the first
# reply is instant with no runtime download. Skipped when local TTS is off.
RUN if [ "$INSTALL_LOCAL_TTS" = "true" ]; then \
        /app/.venv/bin/python -c "from kokoro import KPipeline; p = KPipeline(lang_code='a'); list(p('Gia is ready.', voice='af_heart'))"; \
    fi

COPY . .

ENV PYTHONPATH=/app
# Put the venv on PATH so binaries (alembic, uvicorn) work without "uv run"
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
