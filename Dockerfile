FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "--no-dev", "python", "-m", "worker.entrypoints.worker"]
