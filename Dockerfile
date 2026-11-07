FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

COPY model.py serving.py history.py predict.py streamlit_app.py docker_healthcheck.py ./
COPY .streamlit/ ./.streamlit/
COPY models/ ./models/
COPY data/processed/rossmann.parquet ./data/processed/rossmann.parquet

EXPOSE 9696 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python docker_healthcheck.py

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
