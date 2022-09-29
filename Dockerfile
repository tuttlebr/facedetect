ARG FROM_BASE_IMAGE
FROM ${FROM_BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN curl -s https://deb.nodesource.com/gpgkey/nodesource.gpg.key | apt-key add - && \
    echo 'deb https://deb.nodesource.com/node_18.x focal main' \
    > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y \
    nodejs
RUN pip install pillow~=9.0.1 tritonclient[all] scikit-learn~=0.24.2