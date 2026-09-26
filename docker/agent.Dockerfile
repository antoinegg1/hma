ARG AGENT_BASE_IMAGE
FROM ${AGENT_BASE_IMAGE}
USER root
COPY . /opt/hma-supplement
RUN python3 -m pip install -r /opt/hma-supplement/requirements-runtime.lock \
    && python3 -m pip install --no-deps /opt/hma-supplement
ENV HUMANIZE_SENTRY=off
