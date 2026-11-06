FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

COPY model.py predict.py ./
COPY models/ ./models/

EXPOSE 9696

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9696/health')"

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
