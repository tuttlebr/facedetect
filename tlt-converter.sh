tao-converter /tmp/facenet_model.etlt \
              -k nvidia_tlt \
              -d 3,416,736 \
              -o output_bbox/BiasAdd \
              -w 167108864 \
              -e /models/facenet/1/model.plan

tao-converter /tmp/fpenet_model.etlt \
              -k nvidia_tlt \
              -d 1,80,80 \
              -p input_face_images,1x1x80x80,32x1x80x80,64x1x80x80 \
              -o conv_keypoints_m80 \
              -w 167108864 \
              -e /models/face_descriptor/1/model.plan
