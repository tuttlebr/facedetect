ARG TRITON_VERSION=23.02
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
FROM ${BASE_IMAGE} as builder

# Don't prompt on build.
ENV DEBIAN_FRONTEND=noninteractive

# Python dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install --extra-index-url \
    https://developer.download.nvidia.com/compute/redist \
    --upgrade nvidia-dali-cuda120

# Replace libnvinfer_plugin.so since some TRT plugins are not supported in TRT8. 
RUN wget https://nvidia.box.com/shared/static/7u2ocnwenwgrsx1yq8vv4hkfr0dg1rtm -O \
    /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.0.3

# Setting up TensorRT Paths.
ENV TRT_LIB_PATH=/usr/lib/x86_64-linux-gnu
ENV TRT_INC_PATH=/usr/include/x86_64-linux-gnu

# Install protoc as it should bind to whatever version of protobuf is used.
ARG PROTOBUF_URL=https://github.com/protocolbuffers/protobuf/releases/download/v21.6/protoc-21.6-linux-x86_64.zip
RUN wget ${PROTOBUF_URL} -O proto.zip \
    && unzip proto.zip \
    && chmod +x bin/protoc \
    && mv bin/protoc /usr/local/bin

# Download and install TAO Toolkit converter: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter
RUN wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v4.0.0_trt8.5.1.7_x86/zip \
    -O tao-converter_v4.0.0_trt8.5.1.7_x86.zip \
    && unzip tao-converter_v4.0.0_trt8.5.1.7_x86.zip \
    && mv tao-converter /opt \
    && chmod +x /opt/tao-converter

ENV TRT_LIB_PATH=/usr/lib/x86_64-linux-gnu
ENV TRT_INC_PATH=/usr/include/x86_64-linux-gnu
ENV PATH=/opt/tao-converter:$PATH

WORKDIR /opt/tritonserver