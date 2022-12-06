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

---

### RedisDB for saving inference data

#### Model Overview

```python
class Bbox(JsonModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Face(JsonModel):
    bbox: Bbox
    probability: int
    label: Optional[int] = None
    rotation: Optional[int] = None
    descriptors: Optional[Dict] = None


class Model(JsonModel):
    filename: str = Field(index=True, full_text_search=True)
    faces: Optional[List[Face]] = None
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    portrait: Optional[int] = None

    class Meta:
        database = get_redis_connection()
```

---

### FaceDetect Model

#### Model Overview <a class="anchor" name="model_overview"></a>

The model described in this card detects one or more faces in the given image / video. Compared to the FaceirNet model, this model gives better results on RGB images and smaller faces.

#### Model Architecture <a class="anchor" name="model_architecture"></a>

The model is based on NVIDIA DetectNet_v2 detector with ResNet18 as a feature extractor. This architecture, also known as GridBox object detection, uses bounding-box regression on a uniform grid on the input image. Gridbox system divides an input image into a grid which predicts four normalized bounding-box parameters (xc, yc, w, h) and confidence value per output class.

The raw normalized bounding-box and confidence detections needs to be post-processed by a clustering algorithm such as DBSCAN or NMS to produce final bounding-box coordinates and category labels. The results are then saved to your local RedisDB. `docker/triton-server/facenet_model/README.md` for more information.

```sh
ngc registry model download-version "nvidia/tao/facenet:pruned_quantized_v2.0.1" --dest docker/triton-server/facenet_model
```

---

### Facial Landmark Estimator (FPENet) Model Card

#### Model Overview <a class="anchor" name="model_overview"></a>

The FPENet model described in this card is a facial keypoints estimator network, which aims to predict the (x,y) location of keypoints for a given input face image. FPEnet is generally used in conjuction with a face detector and the output is commonly used for face alignment, head pose estimation, emotion detection, eye blink detection, gaze estimation, among others.

This model predicts 68, 80 or 104 keypoints for a given face- Chin: 1-17, Eyebrows: 18-27, Nose: 28-36, Eyes: 37-48, Mouth: 49-61, Inner Lips: 62-68, Pupil: 69-76, Ears: 77-80, additional eye landmarks: 81-104. It can also handle visible or occluded flag for each keypoint. An example of the kaypoints is shown as follows:

<img style="center" src="https://developer.nvidia.com/sites/default/files/akamai/TLT/fpe_sample_keypoints.png" width="500"> <br>

#### Model Architecture <a class="anchor" name="model_architecture"></a>

This is a classification model with a [Recombinator network](https://openaccess.thecvf.com/content_cvpr_2016/papers/Honari_Recombinator_Networks_Learning_CVPR_2016_paper.pdf) backbone. Recombinator networks are a family of CNN architectures that are suited for fine grained pixel level predictions (as oppose to image level prediction like classification). The model recombines the layer inputs such that convolutional layers in the finer branches get inputs from both coarse and fine layers.

The facial landmark estimations may be used for facial alignment and an example of that is provided in the notebook. `docker/triton-server/fpenet_model/README.md` for more information.

```sh
ngc registry model download-version "nvidia/tao/fpenet:deployable_v3.0" --dest docker/triton-server/fpenet_model
```

---

## Containers

### Build Containers

```sh
docker compose build
```

### Run Model Conversion

```sh
docker compose run triton-model-builder
```

### Start Services

```sh
docker compose up triton-server triton-client redis-db
```
