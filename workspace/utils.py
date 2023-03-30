import glob
import os
import random
from multiprocessing import Pool

import numpy as np


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
    formats=(".jpeg", ".jpg", ".png"),
    follow_links=True,
    random_order=False,
):
    """Make list of all files in the subdirs of `directory`, with their labels.
    Args:
      directory: The target directory (string).
      formats: Allowlist of file extensions to index (e.g. ".jpeg", ".jpg", ".png").
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

    if random_order:
        random.shuffle(file_paths)
        return file_paths
    else:
        return sorted(file_paths)


def iter_valid_files(directory, follow_links, formats):
    walk = os.walk(directory, followlinks=follow_links)
    for root, _, files in sorted(walk, key=lambda x: x[0]):
        if not os.path.split(root)[1].startswith("."):
            for fname in sorted(files):
                if fname.lower().endswith(formats):
                    yield root, fname


def parse_descriptors(image_points, scale=(1, 1)):
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
            chin[i] = {
                "x": int(coordinate[0] * scale[0]),
                "y": int(coordinate[1] * scale[1]),
            }
        elif i < 27:
            eyebrows[i] = {
                "x": int(coordinate[0] * scale[0]),
                "y": int(coordinate[1] * scale[1]),
            }
        elif i < 36:
            nose[i] = {
                "x": int(coordinate[0] * scale[0]),
                "y": int(coordinate[1] * scale[1]),
            }
        elif i < 48:
            eyes[i] = {
                "x": int(coordinate[0] * scale[0]),
                "y": int(coordinate[1] * scale[1]),
            }
        elif i < 61:
            mouth[i] = {
                "x": int(coordinate[0] * scale[0]),
                "y": int(coordinate[1] * scale[1]),
            }
        elif i < 68:
            lips[i] = {
                "x": int(coordinate[0] * scale[0]),
                "y": int(coordinate[1] * scale[1]),
            }
        elif i < 76:
            pupil[i] = {
                "x": int(coordinate[0] * scale[0]),
                "y": int(coordinate[1] * scale[1]),
            }
        elif i < 80:
            ears[i] = {
                "x": int(coordinate[0] * scale[0]),
                "y": int(coordinate[1] * scale[1]),
            }
        elif i < 104:
            # Additional eye descriptors.
            eyes[i] = {
                "x": int(coordinate[0] * scale[0]),
                "y": int(coordinate[1] * scale[1]),
            }
        i += 1

    descriptor_points = {
        "chin": chin,
        "eyebrows": eyebrows,
        "nose": nose,
        "eyes": eyes,
        "mouth": mouth,
        "lips": lips,
        "pupil": pupil,
        "ears": ears,
    }
    return descriptor_points


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
