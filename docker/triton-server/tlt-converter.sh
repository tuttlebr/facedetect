#!/bin/bash
MIN_BATCH=1
OPT_BATCH=16
MAX_BATCH=32
# GPU_MEMORY=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits)
# MAX_WORKSPACE_SIZE=$((1<<45))

echo "Preparing FaceNet Models"
echo "----------------------------------------------------------------------------------------------------------------"
echo "[INFO] Serializing FaceNet DALI model"
python3 /tmp/facenet_model/serialize_dali.py
echo "[INFO] FaceNet DALI model serialization complete"
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet/version
# INT8 Calibration
# tao-converter \
#     -k nvidia_tlt \
#     -t int8 \
#     -d 3,416,736 \
#     -i nchw \
#     -e /models/facenet/1/model.trt \
#     -max_workspace_size=${MAX_WORKSPACE_SIZE} \
#     -c /tmp/facenet_model/facenet_vpruned_quantized_v2.0.1/int8_calibration.txt \
#     /tmp/facenet_model/facenet_vpruned_quantized_v2.0.1/model.etlt

# FP32 Calibration
tao-converter \
    -k nvidia_tlt \
    -d 3,416,736 \
    -i nchw \
    -e /models/facenet/1/model.trt \
    -b ${MIN_BATCH} \
    -m ${MAX_BATCH} \
    -p input_1,${MIN_BATCH}x3x416x736,${OPT_BATCH}x3x416x736,${MAX_BATCH}x3x416x736 \
    /tmp/facenet_model/facenet_vdeployable_v1.0/model.etlt


protoc -I=/models/facenet_postprocess/1/postprocessing \
    --python_out=/models/facenet_postprocess/1/postprocessing \
    /models/facenet_postprocess/1/postprocessing/postprocessor_config.proto
echo "[INFO] Completed FaceNet Models prep"

echo
echo "Preparing FPENet Model"
echo "----------------------------------------------------------------------------------------------------------------"
echo "[INFO] Serializing FPENet DALI model"
python3 /tmp/fpenet_model/serialize_dali.py
echo "[INFO] FPENet DALI model serialization complete"
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/fpenet/version
# INT8 Calibration
# tao-converter \
#     -k nvidia_tlt \
#     -t int8 \
#     -o conv_keypoints_m80 \
#     -d 1,80,80 \
#     -i nchw \
#     -e /models/fpenet/1/model.trt \
#     -max_workspace_size=${MAX_WORKSPACE_SIZE} \
#     -p input_face_images,${MIN_BATCH}x1x80x80,${OPT_BATCH}x1x80x80,${MAX_BATCH}x1x80x80 \
#     -c /tmp/fpenet_model/fpenet_vdeployable_v3.0/int8_calibration.txt \
#     /tmp/fpenet_model/fpenet_vdeployable_v3.0/model.etlt

# FP32 Calibration
tao-converter \
    -k nvidia_tlt \
    -o conv_keypoints_m80 \
    -d 1,80,80 \
    -i nchw \
    -e /models/fpenet/1/model.trt \
    -max_workspace_size=${MAX_WORKSPACE_SIZE} \
    -p input_face_images,${MIN_BATCH}x1x80x80,${OPT_BATCH}x1x80x80,${MAX_BATCH}x1x80x80 \
    /tmp/fpenet_model/fpenet_vdeployable_v3.0/model.etlt