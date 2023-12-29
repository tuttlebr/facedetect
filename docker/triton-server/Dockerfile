ARG TRITON_VERSION=23.02
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
FROM ${BASE_IMAGE} as builder

# Don't prompt on build.
ENV DEBIAN_FRONTEND=noninteractive

# Python dependencies.
# https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html#nvidia-dali
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install --extra-index-url \
    https://developer.download.nvidia.com/compute/redist \
    --upgrade nvidia-dali-cuda120

# Install protoc as it should bind to whatever version of protobuf is used.
ARG PROTOBUF_URL=https://github.com/protocolbuffers/protobuf/releases/download/v21.6/protoc-21.6-linux-x86_64.zip
RUN wget ${PROTOBUF_URL} -O proto.zip \
    && unzip proto.zip \
    && chmod +x bin/protoc \
    && mv bin/protoc /usr/local/bin

# Download and install TAO Toolkit converter: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter
ARG TAO_VERSION=v5.1.0_8.6.3.1_x86
RUN wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/${TAO_VERSION}/zip \
    -O tao-converter_${TAO_VERSION}.zip \
    && unzip tao-converter_${TAO_VERSION}.zip \
    && mv tao-converter /opt \
    && chmod +x /opt/tao-converter

ENV TRT_LIB_PATH=/usr/lib/x86_64-linux-gnu
ENV TRT_INC_PATH=/usr/include/x86_64-linux-gnu
ENV PATH=/opt/tao-converter:$PATH

WORKDIR /opt/tritonserver
