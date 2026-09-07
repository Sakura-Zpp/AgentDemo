FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANGGRAPH_STRICT_MSGPACK=true

WORKDIR /app

RUN groupadd --system agent && useradd --system --gid agent --create-home agent

COPY pyproject.toml README.md ./
COPY agent ./agent
COPY model ./model
COPY rag ./rag
COPY utils ./utils
COPY config ./config
COPY prompts ./prompts
COPY data ./data
COPY chroma_db ./chroma_db
COPY sha256.text ./sha256.text
COPY app.py healthcheck.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[postgres]" \
    && mkdir -p /app/.agent_data /app/log /app/chroma_db \
    && chown -R agent:agent /app

USER agent
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "healthcheck.py"]

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.fileWatcherType=none"]
