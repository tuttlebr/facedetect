ARG FROM_BASE_IMAGE
FROM ${FROM_BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN curl -s https://deb.nodesource.com/gpgkey/nodesource.gpg.key | apt-key add - && \
    echo 'deb https://deb.nodesource.com/node_18.x focal main' \
    > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y \
    cmake \
    nodejs

# RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade \
    pillow~=9.2.0 \
    tritonclient[all]~=2.25.0 \
    scikit-learn~=0.24.0 \
    pillow_heif~=0.7.0 \
    dlib~=19.24.0 \
    ipywidgets~=8.0.0