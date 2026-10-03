FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential ca-certificates git git-lfs libgomp1 libgl1 libglib2.0-0 ffmpeg unzip p7zip-full && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" HUMANIZE_SENTRY=off
COPY requirements-repro.lock /tmp/requirements-repro.lock
COPY docker/requirements-ml.lock /tmp/requirements-ml.lock
RUN pip install --no-cache-dir -r /tmp/requirements-repro.lock -r /tmp/requirements-ml.lock
COPY . /opt/hma-source
RUN pip install --no-cache-dir --no-deps /opt/hma-source && pip freeze > /opt/hma-environment.freeze.txt
WORKDIR /home/user/workspace
