from __future__ import annotations

import glob
import os
from functools import partial, singledispatch
from multiprocessing import Pool
import time
from typing import Dict, List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from PIL import Image, ImageDraw, ImageFile
from redis_om import Field, JsonModel, Migrator, get_redis_connection


from dotenv import load_dotenv


load_dotenv()

ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL_VERSION = os.getenv("MODEL_VERSION")
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL")
CONTAINER_IMAGE_FOLDER = os.getenv("CONTAINER_IMAGE_FOLDER")
JSON_DATA_FILE = os.getenv("JSON_DATA_FILE")
FACE_DETECT_MODEL_NAME = os.getenv("FACE_DETECT_MODEL_NAME")
FPENET_MODEL_NAME = os.getenv("FPENET_MODEL_NAME")

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
    probability: float
    label: Optional[int] = None
    rotation: Optional[int] = None
    descriptors: Optional[Dict] = None
    face_vector: Optional[List[float]] = None
    center: Optional[List[int]] = None


class Model(JsonModel):
    filename: str = Field(index=True, full_text_search=True)
    faces: Optional[List[Face]] = None
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    portrait: Optional[int] = None

    class Meta:
        database = get_redis_connection()


def render_image(
    model,
    outline_color=(255, 0, 0),
    linewidth=20,
    output_size=None,
):
    """Render images with overlain outputs."""
    image = Image.open(model.filename).convert("RGB")
    if_rotate = 90 * model.portrait
    image = image.rotate(if_rotate, expand=1)
    w, h = model.width, model.height
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


def get_fpenet_rotation(points):
    """This model predicts 68, 80 or 104 keypoints for a given face-
        Chin: 0-16
        Eyebrows: 17-26
        Nose: 27-36
        Eyes: 36-47
        Mouth: 48-60
        Inner Lips: 61-67
        Pupil: 68-75
        Ears: 76-79
        additional eye landmarks: 80-103
        It can also handle visible or occluded flag for each keypoint."""

    try:
        left = points[36:41]
        right = points[42:47]
        left_xs = sum([i[0] for i in left]) // len(left)
        left_ys = sum([i[1] for i in left]) // len(left)
        right_xs = sum([i[0] for i in right]) // len(right)
        right_ys = sum([i[1] for i in right]) // len(right)
        tan = (right_ys - left_ys) / (right_xs - left_xs)

        rotation = np.degrees(np.arctan(tan))

    except BaseException:
        rotation = 0

    return round(rotation, 0)


def load_model(model):
    image = Image.open(model.filename)
    if_rotate = 90 * model.portrait
    return image.rotate(if_rotate, expand=1)


def crop_and_rotate_clip(model, rotate=0, resize=0):
    image = Image.open(model.filename).convert("RGB")
    if_rotate = 90 * model.portrait
    image = image.rotate(if_rotate, expand=1)

    faces = []
    for face in model.faces:
        if rotate:
            rotated_image = image.rotate(face.rotation, expand=1)
        else:
            rotated_image = image
        cropped_image = rotated_image.crop(
            (face.bbox.x1,
             face.bbox.y1,
             face.bbox.x2,
             face.bbox.y2))

        if resize:
            cropped_image = cropped_image.resize(
                size=(150, 150), resample=Image.Resampling.LANCZOS)
        faces.append(cropped_image)
    return faces


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


class RedisModelIterator(object):
    def __init__(self, models, batch_size):
        self.models = models
        self.batch_size = batch_size
        self.data_set_len = len(self.models)
        self.n = self.data_set_len

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        model_batch = []
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            model = self.models[self.i % self.n]
            model_batch.append(model)
            self.i += 1
        return model_batch

    def __len__(self):
        return self.data_set_len

    next = __next__


def np_to_triton_dtype(np_dtype):
    if np_dtype == bool:
        return "BOOL"
    elif np_dtype == np.int8:
        return "INT8"
    elif np_dtype == np.int16:
        return "INT16"
    elif np_dtype == np.int32:
        return "INT32"
    elif np_dtype == np.int64:
        return "INT64"
    elif np_dtype == np.uint8:
        return "UINT8"
    elif np_dtype == np.uint16:
        return "UINT16"
    elif np_dtype == np.uint32:
        return "UINT32"
    elif np_dtype == np.uint64:
        return "UINT64"
    elif np_dtype == np.float16:
        return "FP16"
    elif np_dtype == np.float32:
        return "FP32"
    elif np_dtype == np.float64:
        return "FP64"
    elif np_dtype == np.object_ or np_dtype.type == np.bytes_:
        return "BYTES"
    return None


def triton_to_np_dtype(dtype):
    if dtype == "BOOL":
        return bool
    elif dtype == "INT8":
        return np.int8
    elif dtype == "INT16":
        return np.int16
    elif dtype == "INT32":
        return np.int32
    elif dtype == "INT64":
        return np.int64
    elif dtype == "UINT8":
        return np.uint8
    elif dtype == "UINT16":
        return np.uint16
    elif dtype == "UINT32":
        return np.uint32
    elif dtype == "UINT64":
        return np.uint64
    elif dtype == "FP16":
        return np.float16
    elif dtype == "FP32":
        return np.float32
    elif dtype == "FP64":
        return np.float64
    elif dtype == "BYTES":
        return np.object_
    return None


class TritonClient:
    def __init__(
            self,
            model_name: str,
            model_version: str,
            url,
            verbose=False,
            concurrency=1,
            max_batch_size=32,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.url = url
        self.client = grpcclient.InferenceServerClient(
            url=url, verbose=verbose)
        self.model_metadata = self.client.get_model_metadata(
            model_name=self.model_name, model_version=self.model_version
        )
        self.model_config = self.client.get_model_config(
            model_name=self.model_name, model_version=self.model_version
        ).config
        self.input_names = [i.name for i in self.model_config.input]
        self.output_names = [i.name for i in self.model_config.output]
        self._get_input = (
            self._get_facedetect_input
            if self.model_name == FACE_DETECT_MODEL_NAME
            else self._get_fpenet_input
        )
        self.concurrency = concurrency
        self.max_batch_size = self.model_config.max_batch_size

    @staticmethod
    def _get_facedetect_input(
            models_batch,
            batch_size,
            names=["input_image_data"],
            **kwargs):
        files = [model.filename for model in models_batch]
        request_ids = [model.pk for model in models_batch]
        pipe = Pipeline(batch_size=batch_size, num_threads=4,
                        device_id=None, **kwargs)
        inputs = []
        with pipe:
            images, _ = fn.readers.file(files=files, random_shuffle=False)
            images = fn.reshape(images, shape=[1, -1])
            pipe.set_outputs(images)
        pipe.build()
        tensor_lists = pipe.run()[0]
        for tensor_list in tensor_lists:
            image = np.array(tensor_list, dtype=np.uint8)
            inputs.append([grpcclient.InferInput(
                names[0], image.shape, np_to_triton_dtype(image.dtype))])
            inputs[-1][0].set_data_from_numpy(image)
        return inputs, request_ids

    @staticmethod
    def _get_fpenet_input(
            models_batch,
            batch_size,
            names=[
                "raw_image_data",
                "true_boxes"],
            **kwargs):

        files = [model.filename for model in models_batch]
        request_ids = [model.pk for model in models_batch]
        pipe = Pipeline(batch_size=batch_size, num_threads=4,
                        device_id=None, **kwargs)
        inputs = []
        request_ids = []
        with pipe:
            images, _ = fn.readers.file(files=files, random_shuffle=False)
            images = fn.reshape(images, shape=[1, -1])
            pipe.set_outputs(images)
        pipe.build()
        tensor_lists = pipe.run()[0]
        for tidx, tensor_list in enumerate(tensor_lists):
            image = np.array(tensor_list, dtype=np.uint8)
            i = 0
            for face in models_batch[tidx].faces:
                request_ids.append("{}-{}".format(models_batch[tidx].pk, i))
                i += 1
                bboxes = np.expand_dims(
                    np.asarray(
                        (face.bbox.x1,
                         face.bbox.y1,
                         face.bbox.x2,
                         face.bbox.y2),
                        dtype="int32",
                    ),
                    axis=0,
                )

                inputs.append(
                    [
                        grpcclient.InferInput(
                            names[0],
                            image.shape,
                            np_to_triton_dtype(image.dtype),
                        ),
                        grpcclient.InferInput(
                            names[1], bboxes.shape, np_to_triton_dtype(
                                bboxes.dtype)
                        ),
                    ]
                )
                inputs[-1][0].set_data_from_numpy(image)
                inputs[-1][1].set_data_from_numpy(bboxes)
        return inputs, request_ids

    def callback(self, user_data, result, error):
        if error:
            user_data.append(error)
        else:
            user_data.append(result)

    def test_infer(self, models_batch):
        assert len(models_batch) == self.max_batch_size
        inputs, request_ids = self._get_input(
            models_batch, self.max_batch_size, self.input_names)
        outputs = [grpcclient.InferRequestedOutput(
            name) for name in self.output_names]
        async_requests = []
        for input, request_id in zip(inputs, request_ids):
            self.client.async_infer(
                model_name=self.model_name,
                inputs=input,
                callback=partial(self.callback, async_requests),
                outputs=outputs,
                request_id=request_id,
            )
        while len(async_requests) != len(inputs):
            time.sleep(0.2)

        return async_requests
