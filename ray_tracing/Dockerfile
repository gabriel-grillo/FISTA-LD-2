# Para placas mais antigas, você pode tentar usar uma imagem com
# versões mais antigas do CUDA e Ubuntu, por exemplo 11.8.0 e 22.04,
# respectivamente.

ARG CUDAVERSION=13.3.0
ARG UBUNTUVERSION=24.04
ARG BASE_IMAGE=nvidia/cuda:${CUDAVERSION}-devel-ubuntu${UBUNTUVERSION}
FROM ${BASE_IMAGE}

# Installing ray tracing
WORKDIR /ray_tracing
COPY . /ray_tracing
ARG COMP_CAP=120
RUN echo "Using COMP_CAP=${COMP_CAP}" && \
    make COMP_CAP=$COMP_CAP
ENV PYTHONPATH=/ray_tracing/tmp

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    apt-get install -y python3-venv

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install jupyter torch matplotlib tensorflow[and-cuda]
