#!/bin/bash
MIN_BATCH=1
OPT_BATCH=32
MAX_BATCH=64

echo "Preparing FaceNet Models"
echo "----------------------------------------------------------------------------------------------------------------"
echo "[INFO] Serializing FaceNet DALI model"
python3 /tmp/facenet_model/serialize_dali.py
echo "[INFO] FaceNet DALI model serialization complete"
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet/version
# INT8 Calibration - speed > accuracy
tao-converter \
    -k nvidia_tlt \
    -t int8 \
    -b ${MIN_BATCH} \
    -m ${MAX_BATCH} \
    -o output_cov/Sigmoid,output_bbox/BiasAdd \
    -d 3,416,736 \
    -i nchw \
    -c /tmp/facenet_model/facenet_vpruned_quantized_v2.0.1/int8_calibration.txt \
    -e /models/facenet/1/model.trt \
    -p input_1,${MIN_BATCH}x3x416x736,${OPT_BATCH}x3x416x736,${MAX_BATCH}x3x416x736 \
    /tmp/facenet_model/facenet_vpruned_quantized_v2.0.1/model.etlt

#FP32 Calibration speed < accuracy
# tao-converter \
#     -k nvidia_tlt \
#     -t fp32 \
#     -b ${MIN_BATCH} \
#     -m ${MAX_BATCH} \
#     -o output_cov/Sigmoid,output_bbox/BiasAdd \
#     -d 3,416,736 \
#     -i nchw \
#     -e /models/facenet/1/model.trt \
#     -p input_1,${MIN_BATCH}x3x416x736,${OPT_BATCH}x3x416x736,${MAX_BATCH}x3x416x736 \
#     /tmp/facenet_model/facenet_vpruned_v2.0/model.etlt


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
tao-converter \
    -k nvidia_tlt \
    -t int8 \
    -b 1 \
    -m 64 \
    -o conv_keypoints_m80 \
    -d 1,80,80 \
    -i nchw \
    -c /tmp/fpenet_model/fpenet_vdeployable_v3.0/int8_calibration.txt \
    -e /models/fpenet/1/model.trt \
    -p input_face_images,${MIN_BATCH}x1x80x80,${OPT_BATCH}x1x80x80,${MAX_BATCH}x1x80x80 \
    /tmp/fpenet_model/fpenet_vdeployable_v3.0/model.etlt
