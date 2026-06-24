FROM python:3.12-slim

WORKDIR /app

# Local Kokoro TTS (torch + CUDA wheels + a 327MB model) is a DEV convenience so
# we don't spend ElevenLabs credits while iterating. Production TTS is ElevenLabs
# over HTTP and needs none of it — build prod (and the Celery worker) with
# INSTALL_LOCAL_TTS=false for a lean, torch-free image.
ARG INSTALL_LOCAL_TTS=true
ARG INSTALL_LOCAL_STT=true
# Whisper model baked into the image. large-v3 is multilingual and far better on
# accented English (Nigerian, etc.) than base.en; it runs on the GPU at runtime.
ARG STT_MODEL=large-v3
# Distilled local router classifier (sentence-transformers + sklearn, CPU). Adds
# ~200MB (CPU torch) + the MiniLM encoder; opt-in so the lean image stays lean.
ARG INSTALL_ROUTER_CLASSIFIER=false

# System dependencies:
#   nodejs / npm  — spawn the marcelmarais/spotify-mcp-server over MCP stdio
#                   (it is a Node process; the api container is its parent).
#   espeak-ng     — Kokoro's grapheme→phoneme fallback for out-of-vocabulary words.
#   ffmpeg        — faster-whisper uses it to decode WebM/OGG audio from MediaRecorder.
# python:3.12-slim is Debian trixie, which ships Node 20 (>= the MCP SDK's 18).
RUN apt-get update && apt-get install -y --no-install-recommends \
        nodejs npm espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Generous timeout + retries so a slow PyPI mirror doesn't abort the build
# (the default 15s read timeout flakes on large/slow wheels).
RUN pip install --timeout 120 --retries 10 uv

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

# faster-whisper is installed via direct pip (not uv sync) so the lock file does
# not need to include it. ctranslate2 ships as a manylinux wheel — no build tools
# needed. ffmpeg (above) handles WebM/OGG decoding at transcription time.
RUN if [ "$INSTALL_LOCAL_STT" = "true" ]; then \
        uv pip install --python /app/.venv faster-whisper; \
    fi

# GPU runtime libraries for ctranslate2 (faster-whisper's backend). ctranslate2
# 4.5+ needs CUDA 12 + cuDNN 9; install them as pip wheels so the model runs on
# the host GPU when one is passed through (compose `deploy.devices`) and falls
# back to CPU automatically when it is not — no CUDA base image required.
# ~1.3 GB, so it is gated to the STT image only (not the worker).
RUN if [ "$INSTALL_LOCAL_STT" = "true" ]; then \
        uv pip install --python /app/.venv \
            "nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9.0,<10"; \
    fi

# Distilled router classifier serving. Install CPU-only torch explicitly (the
# default sentence-transformers dep would pull the ~2GB CUDA torch wheel; the
# classifier serves on CPU), then sentence-transformers + sklearn + joblib.
RUN if [ "$INSTALL_ROUTER_CLASSIFIER" = "true" ]; then \
        uv pip install --python /app/.venv torch --index-url https://download.pytorch.org/whl/cpu && \
        uv pip install --python /app/.venv "sentence-transformers>=3.0.0" "scikit-learn>=1.4.0" joblib; \
    fi

# Bake the MiniLM encoder into the image so the first local-route is instant.
RUN if [ "$INSTALL_ROUTER_CLASSIFIER" = "true" ]; then \
        /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"; \
    fi

# Pre-download Kokoro's model + default voice INTO the image (one synth warms
# the 327MB weights + af_heart voice into /root/.cache/huggingface) so the first
# reply is instant with no runtime download. Skipped when local TTS is off.
RUN if [ "$INSTALL_LOCAL_TTS" = "true" ]; then \
        /app/.venv/bin/python -c "from kokoro import KPipeline; p = KPipeline(lang_code='a'); list(p('Gia is ready.', voice='af_heart'))"; \
    fi

# Pre-download the Whisper model into the image so the first transcription is
# instant (no runtime download). large-v3 is ~3 GB; fetched on CPU at build time
# (the GPU isn't visible during build) and loaded on CUDA at runtime from the
# same Hugging Face cache.
RUN if [ "$INSTALL_LOCAL_STT" = "true" ]; then \
        STT_MODEL="$STT_MODEL" /app/.venv/bin/python -c \
        "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['STT_MODEL'], device='cpu', compute_type='int8')"; \
    fi

COPY . .

ENV PYTHONPATH=/app
# Put the venv on PATH so binaries (alembic, uvicorn) work without "uv run"
ENV PATH="/app/.venv/bin:$PATH"
# ctranslate2 dlopen()s cuBLAS + cuDNN from the pip wheels installed above.
# Harmless when STT/GPU is off — the paths simply won't exist and CPU is used.
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"

EXPOSE 8000
