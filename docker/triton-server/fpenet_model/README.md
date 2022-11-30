# Facial Landmark Estimator (FPENet) Model Card

## Model Overview <a class="anchor" name="model_overview"></a>

The FPENet model described in this card is a facial keypoints estimator network, which aims to predict the (x,y) location of keypoints for a given input face image. FPEnet is generally used in conjuction with a face detector and the output is commonly used for face alignment, head pose estimation, emotion detection, eye blink detection, gaze estimation, among others.

This model predicts 68, 80 or 104 keypoints for a given face- Chin: 1-17, Eyebrows: 18-27, Nose: 28-36, Eyes: 37-48, Mouth: 49-61, Inner Lips: 62-68, Pupil: 69-76, Ears: 77-80, additional eye landmarks: 81-104. It can also handle visible or occluded flag for each keypoint. An example of the kaypoints is shown as follows:

<img style="center" src="https://developer.nvidia.com/sites/default/files/akamai/TLT/fpe_sample_keypoints.png" width="500"> <br>

## Model Architecture <a class="anchor" name="model_architecture"></a>

This is a classification model with a [Recombinator network](https://openaccess.thecvf.com/content_cvpr_2016/papers/Honari_Recombinator_Networks_Learning_CVPR_2016_paper.pdf) backbone. Recombinator networks are a family of CNN architectures that are suited for fine grained pixel level predictions (as oppose to image level prediction like classification). The model recombines the layer inputs such that convolutional layers in the finer branches get inputs from both coarse and fine layers.

## Training Algorithmn <a class="anchor" name="training_algorithm"></a>

This model was trained using the FPENet entrypoint in TAO. The training algorithm optimizes the network to minimize the manhattan distance (L1), squared euclidean (L2) or the Wing Loss over the keypoints. Individual face regions can be weighted based on- the 'eyes', the 'mouth', the 'pupil' and the rest of the 'face'.

### Training Data and Ground-truth Labeling Guidelines

A pre-trained (`trainable`) model is available, trained on a combination of NVIDIA internal dataset and [Multi-PIE dataset](http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html). NVIDIA internal data has approximately 500k images and Multipie has 750k images.

The ground truth dataset is created by labeling ground-truth facial keypoints by human labellers.

If you are looking to re-train with your own dataset, please follow the guideline below.

- Label the keypoints in the correct order as accuractely as possible. The human labeler would be able to zoom in to a face region to correctly localize the keypoint.
- For keypoints that are not easily distinguishable such as chin or nose, the best estimate should be made by the human labeler. Some keypoints are easily distinguishable such as mouth corners or eye corners.
- Label a keypoint as "occluded" if the keypoint is not visible due to an external object or due to extreme head pose angles. A keypoint is considered occluded when the keypoint is in the image but not visible.
- To reduce discrepency in labeling between multiple human labelers, the same keypoint ordering and instructions should be used across labelers. An independent human labeler may be used to test the quality of the annotated landmarks and potential corrections.

Face bounding boxes labeling:

- Face bounding boxes should be as tight as possible.
- Label each face bounding box with an occlusion level ranging from 0 to 9. 0 means the face is fully visible and 9 means the face is 90% or more occluded. For training, only faces with occlusion level 0-5 are considered.
- The datasets consist of webcam images so truncation is rarely seen. If faces are at the edge of the frame with visibility less than 60% due to truncation, this image is dropped from the dataset.

The [Sloth](https://github.com/cvhciKIT/sloth) and [Label-Studio](https://labelstud.io/) tools have been utilized for labeling.

## Performance

### Evaluation Dataset

The evaluation is done on the [Multi-PIE dataset](http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html)
Users IDs that are used for KPI- 342 079 164 250 343 080 165 251 344 081 166 252 345 082 167 253 346 083 168 254 084 169 255

### Methodology and KPI accuracy <a class="anchor" name="kpi_accuracy"></a>

The region keypoint pixel error is the mean euclidean error in pixel location prediction as compared to the ground truth. We bucketize and average the error per face region (eyes, mouth, chin, etc.).
Metric- Region keypoints pixel error

- All keypoints: 6.1
- Eyes region: 3.33
- Mouth region: 2.96

### Average latency <a class="anchor" name="average_latency"></a>

- Batch Size = 1 at FP16
- T4 - 0.33 ms
- Jetson AGX - 0.84 ms
- Jetson NX - 1.3 ms
  (all measurements using `trtexec` on the specific hardware)

### Real-time Inference Performance <a class="anchor" name="realtime_inference_performance"></a>

The inference uses FP16 precision. The inference performance runs with [`trtexec`](https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/trtexec) on Jetson Nano, AGX Xavier, Xavier NX and NVIDIA T4 GPU. The Jetson devices run at Max-N configuration for maximum system performance. The end-to-end performance with streaming video data might slightly vary depending on use cases of applications.

| Device | Precision | Batch_size | FPS  |
| ------ | --------- | ---------- | ---- |
| Nano   | FP16      | 1          | 115  |
| NX     | FP16      | 1          | 483  |
| Xavier | FP16      | 1          | 1015 |
| T4     | FP16      | 1          | 2489 |

## How to use this model <a class="anchor" name="how_to_use_this_model"></a>

This model needs to be used with NVIDIA Hardware and Software. For Hardware, the model can run on any NVIDIA GPU including NVIDIA Jetson devices. This model can only be used with [Train Adapt Optimize (TAO) Toolkit](https://developer.nvidia.com/tao-toolkit), [DeepStream 6.0](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps) or [TensorRT](https://developer.nvidia.com/tensorrt).

There are two flavors of the model:

- trainable
- deployable

The `trainable` model is intended for training using TAO Toolkit and the user's own dataset. This can provide high fidelity models that are adapted to the use case. The Jupyter notebook available as a part of [TAO container](https://ngc.nvidia.com/catalog/containers/nvidia:tao:tao-toolkit-tf) can be used to re-train.
The `deployable` model is intended for efficient deployment on the edge using DeepStream or [TensorRT](https://developer.nvidia.com/tensorrt).
The `trainable` and `deployable` models are encrypted and will only operate with the following key:

- Model load key: `nvidia_tlt`

Please make sure to use this as the key for all TAO commands that require a model load key.

### Input

Images of 80 X 80 X 1

### Output

N X 2 keypoint locations.
N X 1 keypoint confidence.

N is the number of keypoints.

### General purpose key point estimation

Besides predicting the 68, 80 points, this model can be finetuned to predict other number of facial points or general purpose key points with TAO toolkit above 22.04 version. Following is an example to enable 10 keypoints estimation by changing `num_keypoints` in the training specification file:

```yaml
num_keypoints: 10
dataloader:
  ...
  num_keypoints: 10
  ...
```

## Limitations <a class="anchor" name="limitations"></a>

Some known limitations include relative increase in keypoint estimation error in extreme head pose (yaw > 60 degree) and occlusions.

## Model versions:

- **trainable_v1.0** - Pre-trained model that is intended for training.
- **deployable_v1.0** - Deployment models that is intended to run on the inference pipeline.
- **deployable_v3.0** - Deployment models that is intended to run on the inference pipeline with int8 calibration.

## References <a class="anchor" name="citations"></a>

### Citations <a class="anchor" name="citations"></a>

- Honari, S., Molchanov, P., Tyree, S., Vincent, P., Pal, C., & Kautz, J. (2018). Improving landmark localization with semi-supervised learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 1546-1555).

- Feng, Z. H., Kittler, J., Awais, M., Huber, P., & Wu, X. J. (2018). Wing loss for robust facial landmark localisation with convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2235-2245).

## Using TAO Pre-trained Models <a class="anchor" name="using_tao_pretrained_models"></a>

- Get [TAO Container](https://ngc.nvidia.com/catalog/containers/nvidia:tao:tao-toolkit-tf)
- Get other Purpose-built models from NGC model registry:
  - [FaceNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:facenet)
  - [FPENet](https://ngc.nvidia.com/catalog/models/nvidia:tao:fpenet)
  - [GazeNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:gazenet)
  - [EmotionNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:emotionnet)
  - [GestureNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:gesturenet)
  - [HeartRateNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:heartratenet)
  - [PeopleNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplenet)
  - [TrafficCamNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:trafficcamnet)
  - [DashCamNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:dashcamnet)
  - [VehicleMakeNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:vehiclemakenet)
  - [VehicleTypeNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:vehicletypenet)

## Technical blogs <a class="anchor" name="technical_blogs"></a>

- Read the 2 part blog on training and optimizing 2D body pose estimation model with TAO - [Part 1](https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-1) | [Part 2](https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-2)
- Read the technical tutorial on how [PeopleNet model can be trained with custom data using Transfer Learning Toolkit](https://devblogs.nvidia.com/training-custom-pretrained-models-using-tlt/)

## Suggested reading <a class="anchor" name="suggested_reading"></a>

- More information on about TAO Toolkit and pre-trained models can be found at the [NVIDIA Developer Zone](https://developer.nvidia.com/tao-toolkit)
- Read the [TAO getting Started](https://docs.nvidia.com/tao/tao-toolkit/text/tao_quick_start_guide.html) guide and [release notes](https://docs.nvidia.com/tao/tao-toolkit/text/release_notes.html).
- If you have any questions or feedback, please refer to the discussions on [TAO Toolkit Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/tao-toolkit/17)
- Deploy your model on the edge using DeepStream. Learn more about [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)

## License <a class="anchor" name="license"></a>

License to use this model is covered by the Model EULA. By downloading the unpruned or pruned version of the model, you accept the terms and conditions of these [licenses](https://developer.nvidia.com/deep-learning-models-license-agreement).

## Ethical Considerations <a class="anchor" name="ethical_considerations"></a>

Training and evaluation dataset mostly consists of North American content. An ideal training and evaluation dataset would additionally include content from other geographies.

NVIDIA’s platforms and application frameworks enable developers to build a wide array of AI applications. Consider potential algorithmic bias when choosing or creating the models being deployed. Work with the model’s developer to ensure that it meets the requirements for the relevant industry and use case; that the necessary instruction and documentation are provided to understand error rates, confidence intervals, and results; and that the model is being used under the conditions and in the manner intended.
