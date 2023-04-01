ARG TRITON_VERSION=23.02
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:${TRITON_VERSION}-py3
FROM ${BASE_IMAGE} as builder

# Don't prompt on build.
ENV DEBIAN_FRONTEND=noninteractive

# Python dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install protoc as it should bind to whatever version of protobuf is used.
ARG PROTOBUF_URL=https://github.com/protocolbuffers/protobuf/releases/download/v21.6/protoc-21.6-linux-x86_64.zip
RUN wget ${PROTOBUF_URL} -O proto.zip \
    && unzip proto.zip \
    && chmod +x bin/protoc \
    && mv bin/protoc /usr/local/bin