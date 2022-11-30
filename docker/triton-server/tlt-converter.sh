#!/bin/bash

echo "[INFO] Preparing FaceNet Models"

echo "[INFO] Serializing DALI model"
python3 /tmp/facenet_model/serialize_dali.py
echo "[INFO] DALI model serialization complete"

tao-converter \
    -k nvidia_tlt \
    -t int8 \
    -b 1 \
    -m 32 \
    -o output_cov/Sigmoid,output_bbox/BiasAdd \
    -d 3,416,736 \
    -i nchw \
    -c /tmp/facenet_model/facenet_vpruned_quantized_v2.0.1/int8_calibration.txt \
    -e /models/facenet/1/model.trt \
    -p input_1,1x3x416x736,16x3x416x736,32x3x416x736 \
    /tmp/facenet_model/facenet_vpruned_quantized_v2.0.1/model.etlt

protoc -I=/models/facenet_postprocess/1/postprocessing \
    --python_out=/models/facenet_postprocess/1/postprocessing \
    /models/facenet_postprocess/1/postprocessing/postprocessor_config.proto
    
echo "[INFO] Completed FaceNet Models prep"


echo "[INFO] Preparing FPENet Model"

echo "[INFO] Serializing DALI model"
python3 /tmp/fpenet_model/serialize_dali.py
echo "[INFO] DALI model serialization complete"

tao-converter \
    -k nvidia_tlt \
    -t int8 \
    -b 1 \
    -m 32 \
    -o conv_keypoints_m80 \
    -d 1,80,80 \
    -i nchw \
    -c /tmp/fpenet_model/fpenet_vdeployable_v3.0/int8_calibration.txt \
    -e /models/fpenet/1/model.trt \
    -p input_face_images,1x1x80x80,16x1x80x80,32x1x80x80 \
    /tmp/fpenet_model/fpenet_vdeployable_v3.0/model.etlt