# Getting Started with Face Recognition

## Requirements

- Docker
- Docker Compose
- NVIDIA GPUs on host
- NVIDIA AI Enterprise
- NGC CLI

## NVIDIA ML Assets

### NGC CLI

NVIDIA NGC CLI is a command-line interface tool for managing Docker containers in the NVIDIA NGC Registry. With NGC CLI, you can perform the same operations that are available from the NGC website, such as running jobs, viewing ACE and node information, and viewing Docker repositories within your orgs. Read the docs [here](https://docs.ngc.nvidia.com/cli/index.html). You will need a valid API key to download containers, models and other content from the NVIDIA GPU Cloud (NGC).

### FaceNet

Pretrained model provided by NVIDIA

```sh
ngc registry model download-version "nvidia/tao/facenet:pruned_quantized_v2.0.1" --dest facenet_model
```

### FPEnet

```sh
ngc registry model download-version "nvidia/tao/fpenet:deployable_v3.0" --dest fpenet_model
```

### TAO Converter

```sh
ngc registry resource download-version "nvidia/tao/tao-converter:v3.22.05_trt8.2_x86"
```

## Containers

### Build Containers

```sh
docker compose build
```

### Run Model Conversion

```sh
docker compose up tao-converter
```

### Start Triton

```sh
docker compose up triton-server
```

### Run Jupyter Lab

```sh
docker compose up triton-client
```
