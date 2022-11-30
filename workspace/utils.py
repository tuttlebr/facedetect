from __future__ import annotations

import glob
import math
import os
import sys
from functools import partial, singledispatch
from itertools import islice
from multiprocessing import Pool
from timeit import default_timer
from typing import Dict, List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener
from redis_om import Field, JsonModel, Migrator, get_redis_connection

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

from dotenv import load_dotenv


from bbox import *


load_dotenv()
register_heif_opener()

MODEL_VERSION = os.getenv("MODEL_VERSION")
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL")
CONTAINER_IMAGE_FOLDER = os.getenv("CONTAINER_IMAGE_FOLDER")
JSON_DATA_FILE = os.getenv("JSON_DATA_FILE")
FACE_DETECT_MODEL_NAME = os.getenv("FACE_DETECT_MODEL_NAME")
FPENET_MODEL_NAME = os.getenv("FPENET_MODEL_NAME")
N_DEBUG = int(os.getenv("N_DEBUG"))
THREAD_CHUNKS = int(os.getenv("THREAD_CHUNKS"))

triton_client = grpcclient.InferenceServerClient(
    url=TRITON_SERVER_URL, verbose=False)
Migrator().run()


class Bbox(JsonModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Face(JsonModel):
    bbox: Bbox
    probability: int
    label: Optional[int] = None
    rotation: Optional[int] = None
    descriptors: Optional[Dict] = None


class Model(JsonModel):
    filename: str = Field(index=True, full_text_search=True)
    faces: Optional[List[Face]] = None
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None

    class Meta:
        database = get_redis_connection()


def get_rotation(image):
    w, h = image.size
    if_rotate = h > w
    angle = 90 * if_rotate
    return angle


def render_image(
    model,
    outline_color=(255, 0, 0),
    linewidth=20,
    output_size=None,
):
    """Render images with overlain outputs."""
    image = Image.open(model.filename)
    if image.mode != "RGB":
        image = image.convert("RGB")
    angle = get_rotation(image)
    image = image.rotate(angle, expand=1)

    w, h = image.size
    draw = ImageDraw.Draw(image)
    wpercent = 416 / w
    linewidth = int(linewidth / wpercent)
    for face in model.faces:
        box = tuple((face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2))
        if (box[2] - box[0]) >= 0 and (box[3] - box[1]) >= 0:
            draw.rectangle(box, outline=outline_color)
            for i in range(linewidth):
                draw.rectangle(box, outline=outline_color)

    if output_size:
        scale = output_size / w
        scale_w = int(w * scale)
        scale_h = int(h * scale)
        image = image.resize((scale_w, scale_h), Image.Resampling.LANCZOS)

    return image


def render_fpenet_image(clip, points):
    image = Image.fromarray(
        clip.squeeze().transpose().transpose().astype("uint8"), "L"
    ).convert("RGB")

    rotatation = get_fpenet_rotation(points)
    return image.rotate(rotatation, expand=1)


def get_fpenet_rotation(points):
    """This model predicts 68, 80 or 104 keypoints for a given face- Chin: 1-17, Eyebrows: 18-27, Nose: 28-36, Eyes: 37-48, Mouth: 49-61, Inner Lips: 62-68, Pupil: 69-76, Ears: 77-80, additional eye landmarks: 81-104. It can also handle visible or occluded flag for each keypoint."""

    try:
        points_length = len(points)
        if points_length == 104:
            left = points[80:91]
            right = points[92:103]

            left_xs = sum([i[0] for i in left]) // len(left)
            left_ys = sum([i[1] for i in left]) // len(left)
            right_xs = sum([i[0] for i in right]) // len(right)
            right_ys = sum([i[1] for i in right]) // len(right)
            tan = (right_ys - left_ys) / (right_xs - left_xs)

            rotation = np.degrees(np.arctan(tan))

        elif points_length == 80:
            left = points[68:71]
            right = points[72:75]

            left_xs = sum([i[0] for i in left]) // len(left)
            left_ys = sum([i[1] for i in left]) // len(left)
            right_xs = sum([i[0] for i in right]) // len(right)
            right_ys = sum([i[1] for i in right]) // len(right)
            tan = (right_ys - left_ys) / (right_xs - left_xs)

            rotation = np.degrees(np.arctan(tan))

        # elif points_length == 68:
        #     left = points[36:41]
        #     right = points[42:47]

        #     left_xs = sum([i[0] for i in left]) // len(left)
        #     left_ys = sum([i[1] for i in left]) // len(left)
        #     right_xs = sum([i[0] for i in right]) // len(right)
        #     right_ys = sum([i[1] for i in right]) // len(right)
        #     tan = (right_ys - left_ys) / (right_xs - left_xs)

        #     rotation = np.degrees(np.arctan(tan))

        else:
            rotation = 0.0

        return round(rotation, 0)

    except BaseException:
        return 0


def load_model(model):
    return Image.open(model.filename)


def crop_and_rotate_clip(model, rotate=0):
    image = Image.open(model.filename).convert("RGB")
    faces = []
    for face in model.faces:
        if rotate:
            rotated_image = image.rotate(face.rotation, expand=1)
        else:
            rotated_image = image.rotate(0, expand=1)
        cropped_image = rotated_image.crop(
            (face.bbox.x1,
             face.bbox.y1,
             face.bbox.x2,
             face.bbox.y2))
        faces.append(cropped_image)
    return faces


def crop_and_rotate_and_resize_clip(model, size=224):
    image = Image.open(model.filename).convert("RGB")
    angle = get_rotation(image)
    image = image.rotate(angle, expand=1)
    faces = []
    for face in model.faces:
        tmp_image = image.rotate(face.rotation, expand=1).crop(
            (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2)).rotate(
            face.rotation, expand=1).resize(
            (size, size), Image.Resampling.LANCZOS)
        faces.append(tmp_image)
    return faces


def crop_clip(model):
    image = Image.open(model.filename).convert("RGB")
    angle = get_rotation(image)
    image = image.rotate(angle, expand=1)
    faces = []
    for face in model.faces:
        tmp_image = image.crop(
            (face.bbox.x1,
             face.bbox.y1,
             face.bbox.x2,
             face.bbox.y2))
        faces.append(tmp_image)
    return faces


def load_image(image_path):
    """Loads an encoded image as an array of bytes."""
    return np.expand_dims(np.fromfile(image_path, dtype="uint8"), axis=0)


def take(n, iterable):
    """Return first *n* items of the iterable as a list.
        >>> take(3, range(10))
        [0, 1, 2]
    If there are fewer than *n* items in the iterable, all of them are
    returned.
        >>> take(10, range(3))
        [0, 1, 2]
    """
    return list(islice(iterable, n))


def chunked(iterable, n, strict=False):
    """Break *iterable* into lists of length *n*:
        >>> list(chunked([1, 2, 3, 4, 5, 6], 3))
        [[1, 2, 3], [4, 5, 6]]
    By the default, the last yielded list will have fewer than *n* elements
    if the length of *iterable* is not divisible by *n*:
        >>> list(chunked([1, 2, 3, 4, 5, 6, 7, 8], 3))
        [[1, 2, 3], [4, 5, 6], [7, 8]]
    To use a fill-in value instead, see the :func:`grouper` recipe.
    If the length of *iterable* is not divisible by *n* and *strict* is
    ``True``, then ``ValueError`` will be raised before the last
    list is yielded.
    """
    iterator = iter(partial(take, n, iter(iterable)), [])
    if strict:
        if n is None:
            raise ValueError("n must not be None when using strict mode.")

        def ret():
            for chunk in iterator:
                if len(chunk) != n:
                    raise ValueError("iterable is not divisible by n.")
                yield chunk

        return iter(ret())
    else:
        return iterator


def index_subdirectory(directory, follow_links, formats):
    """Recursively walks directory and list image paths and their class index.
    Arguments:
      directory: string, target directory.
      follow_links: boolean, whether to recursively follow subdirectories
        (if False, we only list top-level images in `directory`).
      formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").
    Returns:
      a list of relative file paths
        files.
    """
    dirname = os.path.basename(directory)
    valid_files = iter_valid_files(directory, follow_links, formats)
    filenames = []
    for root, fname in valid_files:
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(
                absolute_path, directory))
        filenames.append(relative_path)
    filenames_trim = [i for i in filenames if r"@" not in i]
    return filenames_trim


def index_directory(
    directory,
    formats=(".jpeg", ".jpg", ".png", ".heic"),
    follow_links=True,
):
    """Make list of all files in the subdirs of `directory`, with their labels.
    Args:
      directory: The target directory (string).
      formats: Allowlist of file extensions to index (e.g. ".jpeg", ".jpg", ".png", ".heic", ".dng").
    Returns:
      file_paths: list of file paths (strings).
    """
    index_start = default_timer()
    subdirs = []
    for subdir in sorted(glob.glob(os.path.join(directory, "*"))):
        if os.path.isdir(os.path.join(directory, subdir)):
            subdirs.append(subdir)
    subdirs = [i for i in subdirs if not i.startswith(".")]

    # Build an index of the files
    # in the different class subfolders.
    pool = Pool()
    results = []
    filenames = []
    for dirpath in (subdir for subdir in subdirs):
        results.append(
            pool.apply_async(
                index_subdirectory,
                (dirpath,
                 follow_links,
                 formats)))

    for res in results:
        partial_filenames = res.get()
        filenames += partial_filenames

    pool.close()
    pool.join()
    file_paths = [os.path.join(directory, fname) for fname in filenames]
    index_end = default_timer()
    runtime = index_end - index_start
    print(
        "Indexed {:,} file(s) in {:.3f} seconds.".format(
            len(file_paths),
            runtime))

    return file_paths


def iter_valid_files(directory, follow_links, formats):
    walk = os.walk(directory, followlinks=follow_links)
    for root, _, files in sorted(walk, key=lambda x: x[0]):
        if not os.path.split(root)[1].startswith("."):
            for fname in sorted(files):
                if fname.lower().endswith(formats):
                    yield root, fname


@singledispatch
def to_serializable_int(val):
    """Used by default."""
    return int(val)


def submit_to_facedetect(
    filename, input_names, output_names, request_id=""
):
    image_data = np.fromfile(filename, dtype='uint8')
    image_data = np.expand_dims(image_data, axis=0)
    inputs = [grpcclient.InferInput(input_names[0], image_data.shape, "UINT8")]
    inputs[0].set_data_from_numpy(image_data)

    outputs = [
        grpcclient.InferRequestedOutput(output_name, class_count=0)
        for output_name in output_names
    ]
    response = triton_client.infer(
        FACE_DETECT_MODEL_NAME, inputs, outputs=outputs, request_id=request_id
    )

    return response


def get_face_clip(image, image_wise_bboxes, c=1, h=80, w=80):
    image_wise_bboxes = np.array([float(i) for i in image_wise_bboxes])
    image = image.crop(
        (image_wise_bboxes)).resize(
        (h, w), Image.Resampling.LANCZOS)
    clip = np.array(image, dtype="float32").reshape((1, c, h, w))

    return clip


def submit_to_fpenet(
    model,
    input_names=["raw_image_data", "true_boxes"],
    output_names=["conv_keypoints_m80", "softargmax", "softargmax:1"],
):
    responses = []
    image_data = np.fromfile(model.filename, dtype='uint8')
    image_data = np.expand_dims(image_data, axis=0)

    for face in model.faces:
        bboxes = np.asarray(
            (face.bbox.x1,
             face.bbox.y1,
             face.bbox.x2,
             face.bbox.y2),
            dtype="int32")
        bboxes = np.expand_dims(bboxes, axis=0)
        inputs = [
            grpcclient.InferInput(
                input_names[0],
                image_data.shape,
                "UINT8"),
            grpcclient.InferInput(
                input_names[1],
                bboxes.shape,
                "INT32")]
        inputs[0].set_data_from_numpy(image_data)
        inputs[1].set_data_from_numpy(bboxes)

        outputs = [
            grpcclient.InferRequestedOutput(output_name, class_count=0)
            for output_name in output_names
        ]

        response = triton_client.infer(
            FPENET_MODEL_NAME, inputs, outputs=outputs, request_id=model.pk
        )
        responses.append(response)
    return responses


def parse_descriptors(image_points):
    i = 0
    chin = {}
    eyebrows = {}
    nose = {}
    eyes = {}
    mouth = {}
    lips = {}
    pupil = {}
    ears = {}

    for coordinate in image_points:
        if i < 17:
            chin[i] = {"x": int(coordinate[0]), "y": int(coordinate[1])}
        elif i < 27:
            eyebrows[i] = {"x": int(coordinate[0]), "y": int(coordinate[1])}
        elif i < 36:
            nose[i] = {"x": int(coordinate[0]), "y": int(coordinate[1])}
        elif i < 48:
            eyes[i] = {"x": int(coordinate[0]), "y": int(coordinate[1])}
        elif i < 61:
            mouth[i] = {"x": int(coordinate[0]), "y": int(coordinate[1])}
        elif i < 68:
            lips[i] = {"x": int(coordinate[0]), "y": int(coordinate[1])}
        elif i < 76:
            pupil[i] = {"x": int(coordinate[0]), "y": int(coordinate[1])}
        elif i < 80:
            ears[i] = {"x": int(coordinate[0]), "y": int(coordinate[1])}
        elif i < 104:
            # Additional eye descriptors.
            eyes[i] = {"x": int(coordinate[0]), "y": int(coordinate[1])}
        i += 1

    descriptor_points = {"chin": chin,
                         "eyebrows": eyebrows,
                         "nose": nose,
                         "eyes": eyes,
                         "mouth": mouth,
                         "lips": lips,
                         "pupil": pupil,
                         "ears": ears
                         }
    return descriptor_points


def new_rotated_bbox(model):
    image = Image.open(model.filename)
    angle = get_rotation(image)
    image = image.rotate(angle, expand=1)
    w = model.width
    h = model.height
    cx, cy = w // 2, h // 2

    for i, face in enumerate(model.faces):
        bboxes = np.array(
            [(face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2)], dtype="float32")
        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:, 4:]))
        img_rotated = image.rotate(face.rotation)
        corners[:, :8] = rotate_box(
            corners[:, :8], face.rotation, cx, cy, h, w)
        new_bbox = get_enclosing_box(corners)
        scale_factor_x = img_rotated.size[0] / w
        scale_factor_y = img_rotated.size[1] / h

        img_rotated = img_rotated.resize((w, h))

        new_bbox[:,
                 :4] /= [scale_factor_x,
                         scale_factor_y,
                         scale_factor_x,
                         scale_factor_y]
        bboxes = new_bbox
        bboxes = clip_box(bboxes, [0, 0, w, h], 0.25).squeeze()

        model.faces[i].bbox.x1 = int(round(bboxes[0], 0))
        model.faces[i].bbox.y1 = int(round(bboxes[1], 0))
        model.faces[i].bbox.x2 = int(round(bboxes[2], 0))
        model.faces[i].bbox.y2 = int(round(bboxes[3], 0))

    return model
