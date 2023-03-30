"""Utitilies to handle KITTI label file."""

import os

from postprocessing.types import KittiBbox


def write_kitti_annotation(label_file, objects):
    """Write a kitti annotation file."""
    if not os.path.exists(os.path.dirname(label_file)):
        raise NotFoundError(
            "Label file cannot be written to dir: {}".format(
                os.path.dirname(label_file)
            )
        )
    assert isinstance(objects, list), "The annotation must be a list of objects." ""
    with open(label_file, "w") as lfile:
        for label in objects:
            if not isinstance(label, KittiBbox):
                raise NotImplementedError("Cannot serialize label object")
            lfile.write("{}\n".format(str(label)))
    return lfile.closed
