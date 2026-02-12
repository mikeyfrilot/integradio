FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY data_vis/ data_vis/
COPY data_vis_app/ data_vis_app/
COPY rxconfig.py ./
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir ".[all]" reflex

FROM python:3.11-slim
RUN groupadd -r integradio && useradd -r -g integradio integradio
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY --chown=integradio:integradio data_vis/ data_vis/
COPY --chown=integradio:integradio data_vis_app/ data_vis_app/
COPY --chown=integradio:integradio rxconfig.py ./
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 3000
USER integradio
CMD ["reflex", "run", "--env", "prod"]
