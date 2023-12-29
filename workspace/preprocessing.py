import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import psutil
from nvidia.dali import pipeline_def

CORE_COUNT = psutil.cpu_count(logical=False)
GLOBAL_SEED = 42


@pipeline_def(num_threads=CORE_COUNT, seed=GLOBAL_SEED)
def facedetect_pipeline(filenames, shard_id=0, num_shards=1):
    encoded, _ = fn.readers.file(
        files=filenames,
        dont_use_mmap=True,
        read_ahead=True,
        prefetch_queue_depth=CORE_COUNT,
        name="Encoder",
        num_shards=num_shards,
        shard_id=shard_id,
    )

    images = fn.decoders.image(encoded,
                               output_type=types.GRAY,
                               device="mixed",
                               hw_decoder_load=0.95)
    # HWC format is default
    shapes = fn.shapes(images)
    images = fn.color_space_conversion(images,
                                       image_type=types.GRAY,
                                       output_type=types.RGB)
    images = fn.resize(
        images,
        resize_x=736,
        resize_y=416,
        interp_type=types.INTERP_LANCZOS3,
        antialias=True,
    )
    # image * (1/255.)
    images = fn.transpose(images, perm=[2, 0, 1]) * 0.00392156862745098
    return shapes, images, encoded


@pipeline_def(num_threads=CORE_COUNT, seed=GLOBAL_SEED)
def coco_pipeline(coco_annotations_file):
    encoded, bboxes, labels, mask_polygons, mask_vertices = fn.readers.coco(
        annotations_file=coco_annotations_file,
        file_root=".",
        random_shuffle=True,  # Load in no particular order
        skip_empty=True,  # Load only samples with bboxes or polygons
        polygon_masks=True,  # Load segmentation mask data as polygons
        ratio=
        False,  # Bounding box and mask polygons to be expressed in relative coordinates
        ltrb=
        False,  # Bounding boxes to be expressed as left, top, right, bottom coordinates, or not
    )

    images = fn.decoders.image(encoded, device="mixed", hw_decoder_load=0.95)

    # HWC format is default
    fn.shapes(images)
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
