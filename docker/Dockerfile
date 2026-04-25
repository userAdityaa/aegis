FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV AEGIS_APP_MODE=demo
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

WORKDIR /app

COPY pyproject.toml ./
COPY environment ./environment
COPY rewards ./rewards
COPY training ./training
COPY eval ./eval
COPY openenv.yaml ./openenv.yaml
COPY docker ./docker
COPY README.md ./README.md

RUN pip install --upgrade pip \
    && pip install -e .[openenv,server,eval,demo]

EXPOSE 7860

CMD ["sh", "-c", "if [ \"$AEGIS_APP_MODE\" = \"mcp\" ]; then python -m environment.mcp_server; else python docker/demo.py --server-name ${GRADIO_SERVER_NAME:-0.0.0.0} --server-port ${GRADIO_SERVER_PORT:-7860}; fi"]