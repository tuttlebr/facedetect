import logging
from contextlib import contextmanager
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import tritonclient.grpc.model_config_pb2 as mc

logger = logging.getLogger(__name__)


@contextmanager
def pool_context(*args, **kwargs):
    """Simple wrapper to get pool context with close function."""
    pool = Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.terminate()


def denormalize_bounding_bboxes(
    bbox_array,
    stride,
    offset,
    bbox_norm,
    num_classes,
    scale_w,
    scale_h,
    data_format,
    model_shape,
    input_image_size,
    this_id,
):
    """Convert bbox from relative coordinates to absolute coordinates."""

    boxes = deepcopy(bbox_array)
    if data_format == mc.ModelInput.FORMAT_NCHW:
        _, model_height, model_width = model_shape
    else:
        model_height, model_width, _ = model_shape
    scales = np.zeros((boxes.shape[0], 4, boxes.shape[2], boxes.shape[3])).astype(
        np.float32
    )

    for i in range(boxes.shape[0]):
        h, w, _ = input_image_size[i]
        scales[i, 0, :, :].fill(float(w / model_width))
        scales[i, 1, :, :].fill(float(h / model_height))
        scales[i, 2, :, :].fill(float(w / model_width))
        scales[i, 3, :, :].fill(float(h / model_height))
    scales = np.asarray(scales).astype(np.float32)
    target_shape = boxes.shape[-2:]
    gc_centers = [(np.arange(s) * stride + offset) for s in target_shape]
    gc_centers = [s / n for s, n in zip(gc_centers, bbox_norm)]
    for n in range(num_classes):
        boxes[:, 4 * n + 0, :, :] -= gc_centers[0][:, np.newaxis] * scale_w
        boxes[:, 4 * n + 1, :, :] -= gc_centers[1] * scale_h
        boxes[:, 4 * n + 2, :, :] += gc_centers[0][:, np.newaxis] * scale_w
        boxes[:, 4 * n + 3, :, :] += gc_centers[1] * scale_h
        boxes[:, 4 * n + 0, :, :] *= -bbox_norm[0]
        boxes[:, 4 * n + 1, :, :] *= -bbox_norm[1]
        boxes[:, 4 * n + 2, :, :] *= bbox_norm[0]
        boxes[:, 4 * n + 3, :, :] *= bbox_norm[1]
        # Scale back boxes.
        boxes[:, 4 * n + 0, :, :] = (
            np.minimum(np.maximum(boxes[:, 4 * n + 0, :, :], 0), model_width)
            * scales[:, 0, :, :]
        )
        boxes[:, 4 * n + 1, :, :] = (
            np.minimum(np.maximum(boxes[:, 4 * n + 1, :, :], 0), model_height)
            * scales[:, 1, :, :]
        )
        boxes[:, 4 * n + 2, :, :] = (
            np.minimum(np.maximum(boxes[:, 4 * n + 2, :, :], 0), model_width)
            * scales[:, 2, :, :]
        )
        boxes[:, 4 * n + 3, :, :] = (
            np.minimum(np.maximum(boxes[:, 4 * n + 3, :, :], 0), model_height)
            * scales[:, 3, :, :]
        )
    return boxes


def thresholded_indices(cov_array, num_classes, classes, cov_threshold):
    """Threshold out valid bboxes and extract the indices per class."""
    valid_indices = []
    batch_size, num_classes, _, _ = cov_array.shape
    for image_idx in range(batch_size):
        indices_per_class = []
        for class_idx in range(num_classes):
            covs = cov_array[image_idx, class_idx, :, :].flatten()
            class_indices = covs > cov_threshold[classes[class_idx]]
            indices_per_class.append(class_indices)
        valid_indices.append(indices_per_class)
    return valid_indices


def iou_vectorized(rects):
    """
    Intersection over union among a list of rectangles in LTRB format.
    Args:
        rects (np.array) : numpy array of shape (N, 4), LTRB format, assumes L<R and T<B
    Returns::
        d (np.array) : numpy array of shape (N, N) of the IOU between all pairs of rects
    """
    # coordinates
    l, t, r, b = rects.T

    # form intersection coordinates
    isect_l = np.maximum(l[:, None], l[None, :])
    isect_t = np.maximum(t[:, None], t[None, :])
    isect_r = np.minimum(r[:, None], r[None, :])
    isect_b = np.minimum(b[:, None], b[None, :])

    # form intersection area
    isect_w = np.maximum(0, isect_r - isect_l)
    isect_h = np.maximum(0, isect_b - isect_t)
    area_isect = isect_w * isect_h

    # original rect areas
    areas = (r - l) * (b - t)

    # Union area is area_a + area_b - intersection area
    denom = areas[:, None] + areas[None, :] - area_isect

    # Return IOU regularized with .01, to avoid outputing NaN in pathological
    # cases (area_a = area_b = isect = 0)
    return area_isect / (denom + 0.01)
