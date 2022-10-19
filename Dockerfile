ARG FROM_BASE_IMAGE
FROM ${FROM_BASE_IMAGE}

# Don't prompt on build.
ENV DEBIAN_FRONTEND=noninteractive

# Apt update to download packages required to download tao-toolkit.
RUN apt-get update \
    && apt-get install -y \
    cmake \
    libssl-dev \
    unzip

# Copy TAO Toolkit converter
ARG TAO_BINARY_PATH
COPY ${TAO_BINARY_PATH} /usr/local/bin/
RUN chmod +x /usr/local/bin/tao-converter

# Install protoc as it should bind to whatever version of protobuf is used.
ARG PROTOBUF_URL=https://github.com/protocolbuffers/protobuf/releases/download/v21.6/protoc-21.6-linux-x86_64.zip
RUN wget ${PROTOBUF_URL} -O proto.zip \
    && unzip proto.zip \
    && chmod +x bin/protoc \
    && mv bin/protoc /usr/local/bin


# Upgrade pip to help build DLIB et al...
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt