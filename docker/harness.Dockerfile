ARG HMA_AGENT_IMAGE=hma-agent:local
FROM ${HMA_AGENT_IMAGE}
ARG HARNESS
COPY upstream /opt/paper/upstream
COPY paper_runtime /opt/paper/paper_runtime
COPY source.lock.json /opt/paper/source.lock.json
COPY requirements-harness.lock /opt/paper/requirements-harness.lock
COPY model-assets.json /opt/paper/model-assets.json
WORKDIR /opt/paper/upstream
RUN if [ "$HARNESS" = scienceflow ]; then uv sync --frozen --no-dev; \
    elif [ "$HARNESS" = ml_master_v2 ]; then pip install -r /opt/paper/requirements-harness.lock && pip install --no-deps -e .; \
    elif [ "$HARNESS" = mlevolve_no_prior ]; then pip install -r /opt/paper/requirements-harness.lock && python -c "import json; from huggingface_hub import snapshot_download; m=json.load(open('/opt/paper/model-assets.json'))['mlevolve_memory']; snapshot_download(m['repository'], revision=m['revision'], local_dir='/opt/paper/models/bge-base-en-v1.5', ignore_patterns=['onnx/*','openvino/*','*.bin','*.h5','*.msgpack'])"; \
    else exit 1; fi
RUN pip check && pip freeze > /opt/paper/requirements.resolved.txt && chmod 0644 /opt/paper/source.lock.json && chmod -R a+rX /opt/paper /opt/venv && chmod a+x /root && if [ -d /root/.local ]; then chmod -R a+rX /root/.local; fi
ENV PYTHONPATH=/opt/paper HUMANIZE_SENTRY=off
WORKDIR /home/user/workspace
