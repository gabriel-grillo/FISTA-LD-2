ARG CUDAVERSION=13.2.0
ARG UBUNTUVERSION=24.04
ARG BASE_IMAGE=nvidia/cuda:${CUDAVERSION}-devel-ubuntu${UBUNTUVERSION}
FROM ${BASE_IMAGE}

# Installing python and pip
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    apt-get install -y python3-venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Installing requirements from pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir PyWavelets==1.8.0 pydicom==3.0.1 matplotlib==3.10.6 torch==2.8.0 \
    jupyter==1.1.1 tf-nightly==2.23.0.dev20261025 pandas==3.0.5

# Installing texlive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-fonts-recommended \
        texlive-latex-extra \
        cm-super \
        dvipng && \
    rm -rf /var/lib/apt/lists/*

# Installing ray tracing
WORKDIR /ray_tracing
COPY ray_tracing/ .
ARG COMP_CAP=120
RUN echo "Using COMP_CAP=${COMP_CAP}" && \
    make -C /ray_tracing COMP_CAP=$COMP_CAP
ENV PYTHONPATH=/ray_tracing/tmp

# Creating and moving to the working directory
WORKDIR /code

CMD ["/bin/bash"]