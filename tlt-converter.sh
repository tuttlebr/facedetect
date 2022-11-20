#!/bin/bash
echo "---------------------------------------FACENET-----------------------------------------"
tao-converter \
    -k nvidia_tlt \
    -o output_bbox/BiasAdd \
    -d 3,416,736 \
    -i nchw \
    -m 64 \
    -e /models/facenet/1/model.trt \
    -b 32 \
    /tmp/facenet_model/*/model.etlt

protoc -I=/models/facenet_postprocess/1/postprocessing \
    --python_out=/models/facenet_postprocess/1/postprocessing \
    /models/facenet_postprocess/1/postprocessing/postprocessor_config.proto

echo "---------------------------------------FPENET-----------------------------------------"
tao-converter \
    -k nvidia_tlt \
    -o conv_keypoints_m80 \
    -d 1,80,80 \
    -i nchw \
    -m 64 \
    -t int8 \
    -c /tmp/fpenet_model/*/int8_calibration.txt \
    -e /models/fpenet/1/model.trt \
    -b 32 \
    -p input_face_images,1x1x80x80,32x1x80x80,64x1x80x80 \
    /tmp/fpenet_model/*/model.etlt