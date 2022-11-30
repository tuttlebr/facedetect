ARG TRITON_VERSION=22.10
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
FROM ${BASE_IMAGE} as builder

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

# Install NVIDIA DALI Backend
COPY requirements.txt .
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110 \
    && pip install -r requirements.txt

# Install protoc as it should bind to whatever version of protobuf is used.
ARG PROTOBUF_URL=https://github.com/protocolbuffers/protobuf/releases/download/v21.6/protoc-21.6-linux-x86_64.zip
RUN wget ${PROTOBUF_URL} -O proto.zip \
    && unzip proto.zip \
    && chmod +x bin/protoc \
    && mv bin/protoc /usr/local/bin

# Replace libnvinfer_plugin.so since some TRT plugins are not supported in TRT8. 
RUN wget https://nvidia.box.com/shared/static/7u2ocnwenwgrsx1yq8vv4hkfr0dg1rtm -O \
    /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.0.3

# Setting up TensorRT Paths.
ENV TRT_LIB_PATH=/usr/lib/x86_64-linux-gnu
ENV TRT_INC_PATH=/usr/include/x86_64-linux-gnu

# Download and install TAO Toolkit converter
RUN wget https://developer.nvidia.com/tao-converter-80 -P /opt/tao-converter && \
    apt-get update && apt-get install unzip libssl-dev -y && \
    unzip /opt/tao-converter/tao-converter-80 -d /opt/tao-converter && \
    chmod +x /opt/tao-converter/tao-converter-x86-tensorrt8.0/tao-converter

ENV PATH=/opt/tao-converter/tao-converter-x86-tensorrt8.0:$PATH

WORKDIR /opt/tritonserver