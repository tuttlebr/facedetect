#!/bin/bash
MIN_BATCH=8
OPT_BATCH=16
MAX_BATCH=32

echo "[INFO] Preparing FaceNet Models"
echo "[INFO] Serializing FaceNet DALI model"
python3 /tmp/dali/facenet_model/serialize_dali.py
echo "[INFO] FaceNet DALI model serialization complete"

# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet/version
## INT8 Calibration
# /opt/tao-converter \
#     -k nvidia_tlt \
#     -t int8 \
#     -d 3,416,736 \
#     -i nchw \
#     -e /models/facenet/1/model.trt \
#     -c /tmp/facenet_model/int8_calibration.txt \
#     /tmp/facenet_model/model.etlt

## Calibration
/opt/tao-converter \
    -k nvidia_tlt \
    -d 3,416,736 \
    -i nchw \
    -e /models/facenet/1/model.trt \
    -b ${MIN_BATCH} \
    -m ${MAX_BATCH} \
    -p input_1,${MIN_BATCH}x3x416x736,${OPT_BATCH}x3x416x736,${MAX_BATCH}x3x416x736 \
    /tmp/facenet_model/model.etlt


protoc -I=/models/facenet_postprocess/1/postprocessing \
    --python_out=/models/facenet_postprocess/1/postprocessing \
    /models/facenet_postprocess/1/postprocessing/postprocessor_config.proto
echo "[INFO] Completed FaceNet Models prep"
echo "[INFO] Preparing FPENet Model"
echo "[INFO] Serializing FPENet DALI model"
python3 /tmp/dali/fpenet_model/serialize_dali.py
echo "[INFO] FPENet DALI model serialization complete"

# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/fpenet/version
## INT8 Calibration
# /opt/tao-converter \
#     -k nvidia_tlt \
#     -t int8 \
#     -o conv_keypoints_m80 \
#     -d 1,80,80 \
#     -i nchw \
#     -e /models/fpenet/1/model.trt \
#     -p input_face_images,${MIN_BATCH}x1x80x80,${OPT_BATCH}x1x80x80,${MAX_BATCH}x1x80x80 \
#     -c /tmp/fpenet_model/int8_calibration.txt \
#     /tmp/fpenet_model/model.etlt

## Calibration
/opt/tao-converter \
    -k nvidia_tlt \
    -o conv_keypoints_m80 \
    -d 1,80,80 \
    -i nchw \
    -e /models/fpenet/1/model.trt \
    -p input_face_images,${MIN_BATCH}x1x80x80,${OPT_BATCH}x1x80x80,${MAX_BATCH}x1x80x80 \
    /tmp/fpenet_model/model.etlt
