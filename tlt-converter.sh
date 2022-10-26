#!/bin/bash

echo "---------------------------------------FACENET-----------------------------------------"
tao-converter /tmp/facenet_model/*/model.etlt \
    -k nvidia_tlt \
    -d 3,416,736 \
    -o output_bbox/BiasAdd \
    -e /models/facenet/1/model.plan

protoc -I=/models/facenet_postprocess/1/postprocessing \
    --python_out=/models/facenet_postprocess/1/postprocessing \
    /models/facenet_postprocess/1/postprocessing/postprocessor_config.proto

echo "---------------------------------------FPENET-----------------------------------------"
tao-converter /tmp/fpenet_model/*/model.etlt \
    -c /tmp/fpenet_model/*/int8_calibration.txt \
    -k nvidia_tlt \
    -d 1,80,80 \
    -p input_face_images,1x1x80x80,8x1x80x80,16x1x80x80 \
    -o conv_keypoints_m80 \
    -e /models/fpenet/1/model.plan