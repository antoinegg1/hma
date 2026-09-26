ARG EVALUATOR_BASE_IMAGE
FROM ${EVALUATOR_BASE_IMAGE}
USER root
COPY . /opt/hma-supplement
RUN python3 -m pip install '/opt/hma-supplement[evaluation]'
