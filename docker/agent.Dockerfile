FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04
ARG CODEX_VERSION=0.153.0
ARG CLAUDE_CODE_VERSION=2.1.259
ARG KIMI_CODE_VERSION=0.41.0
ARG NODE_VERSION=22.23.2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential ca-certificates curl git git-lfs libgomp1 libgl1 libglib2.0-0 ffmpeg unzip p7zip-full ripgrep xz-utils && rm -rf /var/lib/apt/lists/*
RUN curl -fsSLo /tmp/node.tar.xz "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" && echo 'd60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307  /tmp/node.tar.xz' | sha256sum -c - && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 && rm /tmp/node.tar.xz
RUN npm install -g "@openai/codex@${CODEX_VERSION}" "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" "@moonshot-ai/kimi-code@${KIMI_CODE_VERSION}" && npm cache clean --force
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" HUMANIZE_SENTRY=off PYTHONUNBUFFERED=1
COPY requirements-repro.lock /tmp/requirements-repro.lock
COPY docker/requirements-ml.lock /tmp/requirements-ml.lock
RUN pip install --no-cache-dir -r /tmp/requirements-repro.lock -r /tmp/requirements-ml.lock
COPY . /opt/hma-source
RUN pip install --no-cache-dir --no-deps /opt/hma-source && pip freeze > /opt/hma-environment.freeze.txt && chmod -R a+rX /opt/hma-source /opt/venv
WORKDIR /home/user/workspace
