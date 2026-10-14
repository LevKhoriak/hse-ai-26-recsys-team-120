FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_PYTHON_DOWNLOADS=never
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /workspace
COPY . /workspace
CMD ["bash"]