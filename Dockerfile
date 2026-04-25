FROM python:3.11-slim

RUN useradd -m -u 1000 user

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV AEGIS_APP_MODE=demo
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

USER user
WORKDIR $HOME/app

COPY --chown=user pyproject.toml ./
COPY --chown=user environment ./environment
COPY --chown=user rewards ./rewards
COPY --chown=user training ./training
COPY --chown=user eval ./eval
COPY --chown=user openenv.yaml ./openenv.yaml
COPY --chown=user docker ./docker
COPY --chown=user README.md ./README.md

RUN pip install --upgrade pip \
    && pip install -e .[openenv,server,eval,demo]

EXPOSE 7860

CMD ["sh", "-c", "if [ \"$AEGIS_APP_MODE\" = \"mcp\" ]; then python -m environment.mcp_server; else python docker/demo.py --server-name ${GRADIO_SERVER_NAME:-0.0.0.0} --server-port ${GRADIO_SERVER_PORT:-7860}; fi"]
