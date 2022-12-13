ARG TRITON_VERSION=22.11
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:${TRITON_VERSION}-py3
FROM ${BASE_IMAGE} as builder

# Don't prompt on build.
ENV DEBIAN_FRONTEND=noninteractive

# Python dependencies.
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

# Apt update to download packages required to download tao-toolkit.
RUN apt-get update \
    && apt-get install -y \
    cmake \
    gcc \
    g++ \
    libssl-dev \
    make \
    unzip
    
# add node.js repo and install node.js.
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y nodejs

# Install protoc as it should bind to whatever version of protobuf is used.
ARG PROTOBUF_URL=https://github.com/protocolbuffers/protobuf/releases/download/v21.6/protoc-21.6-linux-x86_64.zip
RUN wget ${PROTOBUF_URL} -O proto.zip \
    && unzip proto.zip \
    && chmod +x bin/protoc \
    && mv bin/protoc /usr/local/bin