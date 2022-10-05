import glob
import os
import sys
from functools import partial, singledispatch
from itertools import islice
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

from dotenv import load_dotenv

load_dotenv()
register_heif_opener()


def render_image(filename, image_wise_bboxes, outline_color=(118, 185, 0), linewidth=3):
    """Render images with overlain outputs."""
    image = Image.open(filename).convert("RGB")
    w, h = image.size
    draw = ImageDraw.Draw(image)
    wpercent = 736 / float(image.size[0])
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
    formats=(".jpeg", ".jpg", ".png", ".heic", ".dng"),
    follow_links=True,
):
    """Make list of all files in the subdirs of `directory`, with their labels.
    Args:
      directory: The target directory (string).
      formats: Allowlist of file extensions to index (e.g. ".jpg", ".png").
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
            pool.apply_async(index_subdirectory, (dirpath, follow_links, formats))
        )
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


def submit_to_facedetect(filename, input_name, output_names, request_id=None):
    image_data = load_image(filename)
    inputs = [grpcclient.InferInput(input_name, image_data.shape, "UINT8")]
    inputs[0].set_data_from_numpy(image_data)

    outputs = [
        grpcclient.InferRequestedOutput(output_name, class_count=0)
        for output_name in output_names
    ]
    return triton_client.infer(
        model_name, inputs, outputs=outputs, request_id=request_id
    )
