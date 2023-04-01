import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import math, pipeline_def


@pipeline_def
def facedetect_pipeline(filenames, shard_id, num_shards):
    encoded, _ = fn.readers.file(
        files=filenames,
        dont_use_mmap=True,
        read_ahead=True,
        prefetch_queue_depth=8,
        name="Encoder",
        num_shards=num_shards,
        shard_id=shard_id,
    )
    # HWC format is default
    shapes = fn.peek_image_shape(encoded)
    image = fn.decoders.image(
        encoded, output_type=types.GRAY, device="mixed", hw_decoder_load=0.5
    )
    image = fn.color_space_conversion(
        image, image_type=types.GRAY, output_type=types.RGB
    )
    image = fn.resize(
        image,
        resize_x=736,
        resize_y=416,
        interp_type=types.INTERP_LANCZOS3,
        antialias=True,
    )
    image = fn.transpose(image, perm=[2, 0, 1]) * (1 / 255.0)
    return shapes, image, encoded


@pipeline_def
def coco_pipeline(coco_annotations_file):
    encoded, bboxes, labels, mask_polygons, mask_vertices = fn.readers.coco(
        annotations_file=coco_annotations_file,
        file_root=".",
        random_shuffle=True,  # Load in no particular order
        skip_empty=True,  # Load only samples with bboxes or polygons
        polygon_masks=True,  # Load segmentation mask data as polygons
        ratio=False,  # Bounding box and mask polygons to be expressed in relative coordinates
        ltrb=False,  # Bounding boxes to be expressed as left, top, right, bottom coordinates
    )
    images = fn.decoders.image(encoded, device="mixed", hw_decoder_load=0.5)
    return images, bboxes, labels, mask_polygons, mask_vertices


def get_face_rotation(points):
    center = points[27]
    left_eye = points[36]
    right_eye = points[45]
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    tan = dY / dX

    # radians to degrees: 180/pi
    rotation = -1 * (np.arctan(tan) * 57.29577951308232)
    return rotation, center
