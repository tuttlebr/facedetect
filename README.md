# Face Recognition with NVIDIA

## Requirements

- Docker
- Docker Compose
- NVIDIA GPUs on host
- NGC CLI
- modifications may be made within the `.env` file to change the images folder to be mounted to the containers as well as many other env options.

## NVIDIA ML Assets

### NGC CLI

NVIDIA NGC CLI is a command-line interface tool for managing Docker containers in the NVIDIA NGC Registry. With NGC CLI, you can perform the same operations that are available from the NGC website, such as running jobs, viewing ACE and node information, and viewing Docker repositories within your orgs. Read the docs [here](https://docs.ngc.nvidia.com/cli/index.html). You will need a valid API key to download containers, models and other content from the NVIDIA GPU Cloud (NGC).

### FaceDetect

Pretrained model provided by NVIDIA
The notebook "workspace/00-facenet-demo.ipynb" will traverse a folder of images and submit those photos to Triton for face finding. The results are then saved to your local RedisDB. `docker/triton-server/facenet_model/README.md` for more information.

```sh
ngc registry model download-version "nvidia/tao/facenet:pruned_quantized_v2.0.1" --dest docker/triton-server/facenet_model
```

### Facial Landmarks Estimation (FPEnet)

Pretrained model provided by NVIDIA
The notebook "workspace/01-fpenet-demo.ipynb" will query your RedisDB for photos with face bounding boxes and submit those to Triton for facial landmark estimation. The results are then saved to your local RedisDB. The facial landmark estimations may be used for facial alignment and an example of that is provided in the notebook. `docker/triton-server/fpenet_model/README.md` for more information.

```sh
ngc registry model download-version "nvidia/tao/fpenet:deployable_v3.0" --dest docker/triton-server/fpenet_model
```

## Containers

### Build Containers

```sh
docker compose build
```

### Run Model Conversion

```sh
docker compose --env-file workspace/.env up tao-converter
```

### Start Services

```sh
docker compose --env-file workspace/.env up triton-server triton-client redis-db redis-insight
```
