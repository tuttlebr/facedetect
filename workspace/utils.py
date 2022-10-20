from __future__ import annotations

import glob
import io
import os
import sys
from functools import partial, singledispatch
from itertools import islice
from multiprocessing import Pool
from timeit import default_timer
from typing import List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener
from redis_om import Field, JsonModel, Migrator, get_redis_connection

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

from dotenv import load_dotenv

load_dotenv()
register_heif_opener()

MODEL_VERSION = os.getenv("MODEL_VERSION")
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL")
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER")
JSON_DATA_FILE = os.getenv("JSON_DATA_FILE")
FACE_DETECT_MODEL_NAME = os.getenv("FACE_DETECT_MODEL_NAME")
FPENET_MODEL_NAME = os.getenv("FPENET_MODEL_NAME")
DLIB_FACE_DESCRIPTOR_MODEL_NAME = os.getenv("DLIB_FACE_DESCRIPTOR_MODEL_NAME")
N_DEBUG = int(os.getenv("N_DEBUG"))
THREAD_CHUNKS = int(os.getenv("THREAD_CHUNKS"))

triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
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
    descriptor: Optional[str] = None


class Model(JsonModel):
    filename: str = Field(index=True, full_text_search=True)
    faces: List[Face]

    class Meta:
        database = get_redis_connection()


def render_image(
    filename,
    image_wise_bboxes,
    outline_color=(118, 185, 0),
    linewidth=10,
    output_size=None,
):
    """Render images with overlain outputs."""
    image = Image.open(filename)
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    draw = ImageDraw.Draw(image)
    wpercent = 736 / w
    linewidth = int(linewidth / wpercent)
    for box in image_wise_bboxes:
        if (box[2] - box[0]) >= 0 and (box[3] - box[1]) >= 0:
            draw.rectangle(box, outline=outline_color)
            for i in range(linewidth):
                x1 = max(0, box[0] - i)
                y1 = max(0, box[1] - i)
                x2 = min(w, box[2] + i)
                y2 = min(h, box[3] + i)
                draw.rectangle(box, outline=outline_color)

    if output_size:
        scale = output_size / w
        scale_w = int(w * scale)
        scale_h = int(h * scale)
        return image.resize((scale_w, scale_h), Image.Resampling.LANCZOS)
    else:
        return image


def load_image(img_path):
    """Loads an encoded image as an array of bytes."""
    return np.expand_dims(np.fromfile(img_path, dtype="uint8"), axis=0)


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
        relative_path = os.path.join(dirname, os.path.relpath(absolute_path, directory))
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
            pool.apply_async(index_subdirectory, (dirpath, follow_links, formats))
        )

    for res in results:
        partial_filenames = res.get()
        filenames += partial_filenames

    pool.close()
    pool.join()
    file_paths = [os.path.join(directory, fname) for fname in filenames]
    index_end = default_timer()
    runtime = index_end - index_start
    print("Indexed {:,} file(s) in {:.3f} seconds.".format(len(file_paths), runtime))

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
    filename, input_names, output_names, request_id="", timeit=False
):
    image = Image.open(filename)
    if image.mode != "RGB":
        image = image.convert("RGB")
    true_image_size = np.array(image).shape
    resized_img = image.resize((736, 416), Image.Resampling.LANCZOS)
    nchw = np.transpose(np.array(resized_img), (2, 0, 1))
    norm = nchw * (1 / 255.0)
    image_data = np.expand_dims(norm, axis=0).astype("float32")
    true_image_size = np.expand_dims(np.array(np.array(image).shape), axis=0).astype(
        "int64"
    )

    inputs = [
        grpcclient.InferInput(input_names[0], image_data.shape, "FP32"),
        grpcclient.InferInput(input_names[1], true_image_size.shape, "INT64"),
    ]
    inputs[0].set_data_from_numpy(image_data)
    inputs[1].set_data_from_numpy(true_image_size)

    outputs = [
        grpcclient.InferRequestedOutput(output_name, class_count=0)
        for output_name in output_names
    ]
    infer_start = default_timer()
    response = triton_client.infer(
        FACE_DETECT_MODEL_NAME, inputs, outputs=outputs, request_id=request_id
    )
    infer_end = default_timer()

    if timeit:
        runtime = index_end - index_start
        print("Inference took {:.3f} seconds.".format(runtime))
    return response


def get_face_clip(img, image_wise_bboxes):
    image_wise_bboxes = np.array([float(i) for i in image_wise_bboxes])
    image = img.crop((image_wise_bboxes)).resize((80, 80), Image.Resampling.LANCZOS)
    fpenet_image = np.array(image, dtype="float32").reshape((1, 1, 80, 80))

    return fpenet_image


def submit_to_fpenet(
    json_items,
    input_name,
    output_names,
    request_id="",
    timeit=False,
    return_clips=False,
):
    responses = []
    clips = []
    image = Image.open(json_items["filename"])
    if image.mode != "L":
        image = image.convert("L")
    infer_start = default_timer()
    for face in json_items["faces"]:
        image_wise_bboxes = tuple(face["bbox"].values())
        fpenet_image = get_face_clip(image, image_wise_bboxes)

        inputs = [grpcclient.InferInput(input_name, fpenet_image.shape, "FP32")]
        inputs[0].set_data_from_numpy(fpenet_image)

        outputs = [
            grpcclient.InferRequestedOutput(output_name, class_count=0)
            for output_name in output_names
        ]

        response = triton_client.infer(
            FPENET_MODEL_NAME, inputs, outputs=outputs, request_id=request_id
        )
        responses.append(response.as_numpy(output_names[1]))
        if return_clips:
            clips.append(fpenet_image)

    infer_end = default_timer()

    if timeit:
        runtime = infer_end - infer_start
        print(
            "Inference took {:.3f} seconds for {:,} faces.".format(
                runtime, len(json_items["faces"])
            )
        )
    if return_clips:
        return responses, clips
    else:
        return responses


def submit_to_dlib(json_items, input_name, output_names, request_id="", timeit=False):
    responses = []
    image = Image.open(json_items["filename"])
    if image.mode != "RGB":
        image = image.convert("RGB")

    infer_start = default_timer()
    for face in json_items["faces"]:
        inputs = []
        outputs = []
        try:
            del face["bbox"]["pk"]
        except:
            continue
        image_wise_bboxes = tuple(face["bbox"].values())
        img_crop = image.crop((image_wise_bboxes))
        img_resize = img_crop.resize((150, 150), Image.Resampling.LANCZOS)
        face_clip = np.expand_dims(np.array(img_resize), axis=0)

        inputs.append(grpcclient.InferInput("face_clip", face_clip.shape, "UINT8"))

        inputs[0].set_data_from_numpy(face_clip)

        outputs = [grpcclient.InferRequestedOutput("face_descriptor")]

        response = triton_client.infer(
            DLIB_FACE_DESCRIPTOR_MODEL_NAME,
            inputs,
            outputs=outputs,
            request_id=request_id,
        )
        responses.append(response)

    infer_end = default_timer()
    if timeit:
        print(
            "Inference took {:.3f} seconds for {:,} faces.".format(
                infer_end - infer_start
            )
        )

    return responses
