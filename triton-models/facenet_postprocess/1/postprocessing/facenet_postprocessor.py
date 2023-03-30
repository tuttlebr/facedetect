import logging
import os

import numpy as np
import postprocessing.postprocessor_config_pb2 as postprocessor_config_pb2
import triton_python_backend_utils as pb_utils
from google.protobuf.text_format import Merge as merge_text_proto
from postprocessing.postprocessor import Postprocessor
from postprocessing.utils import (
    denormalize_bounding_bboxes,
    iou_vectorized,
    thresholded_indices,
)
from sklearn.cluster import DBSCAN as dbscan

logger = logging.getLogger(__name__)


def load_clustering_config(config):
    """Load the clustering config."""
    proto = postprocessor_config_pb2.PostprocessingConfig()

    def _load_from_file(filename, pb2):
        if not os.path.exists(filename):
            raise IOError("Specfile not found at: {}".format(filename))
        with open(filename, "r") as f:
            merge_text_proto(f.read(), pb2)

    _load_from_file(config, proto)
    return proto


class DetectNetPostprocessor(Postprocessor):
    """Post processor for Triton outputs from a DetectNet_v2 client."""

    def __init__(self, data_format, classes, postprocessing_config, target_shape):
        """Initialize a post processor class for a classification model.

        Args:
            output_path (str): Unix path to the output rendered images and labels.
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.
            classes (list): List of the class names.
            postprocessing_config (proto): Configuration elements of the dbscan postprocessor.
            target_shape (tuple): Shape of the model input.
        """
        self.pproc_config = load_clustering_config(postprocessing_config)
        self.classes = classes
        self.output_names = [
            "output_cov/Sigmoid",
            "output_bbox/BiasAdd",
        ]
        self.bbox_norm = [35.0, 35]
        self.offset = 0.5
        self.scale_h = 1
        self.scale_w = 1
        self.target_shape = target_shape
        self.stride = self.pproc_config.stride
        super().__init__(data_format)
        # Format the dbscan elements into classwise configurations for
        # rendering.
        self.configure()

    def configure(self):
        """Configure the post processor object."""
        self.dbscan_elements = {}
        self.coverage_thresholds = {}
        self.box_color = {}
        classwise_clustering_config = self.pproc_config.classwise_clustering_config
        for class_name in self.classes:
            if class_name not in classwise_clustering_config.keys():
                raise KeyError(
                    "Cannot find class name {} in {}".format(
                        class_name, classwise_clustering_config.keys()
                    )
                )

            self.dbscan_elements[class_name] = dbscan(
                eps=classwise_clustering_config[class_name].dbscan_config.dbscan_eps,
                min_samples=classwise_clustering_config[
                    class_name
                ].dbscan_config.dbscan_min_samples,
            )
            self.coverage_thresholds[class_name] = classwise_clustering_config[
                class_name
            ].coverage_threshold
            self.box_color[class_name] = classwise_clustering_config[
                class_name
            ].bbox_color

    def apply(self, results, this_id):
        """Apply the post processing to the outputs tensors.

        This function takes the raw output tensors from the detectnet_v2 model
        and performs the following steps:
        1. Denormalize the output bbox coordinates
        2. Threshold the coverage output to get the valid indices for the bboxes.
        3. Filter out the bboxes from the "output_bbox/BiasAdd" blob.
        4. Cluster the filterred boxes using DBSCAN.
        5. Render the outputs on images and save them to the output_path/images
        """
        output_array = {}
        this_id = int(this_id)
        for output_name in self.output_names:
            request_tensor = pb_utils.get_input_tensor_by_name(results, output_name)
            output_array[output_name] = request_tensor.as_numpy().transpose(0, 1, 3, 2)

        output_array["true_image_size"] = pb_utils.get_input_tensor_by_name(
            results, "true_image_size"
        ).as_numpy()

        abs_bbox = denormalize_bounding_bboxes(
            output_array["output_bbox/BiasAdd"],
            self.stride,
            self.offset,
            self.bbox_norm,
            len(self.classes),
            self.scale_w,
            self.scale_h,
            self.data_format,
            self.target_shape,
            output_array["true_image_size"],
            this_id - 1,
        )

        valid_indices = thresholded_indices(
            output_array["output_cov/Sigmoid"],
            len(self.classes),
            self.classes,
            self.coverage_thresholds,
        )

        batchwise_boxes = []
        batchwise_proba = []
        for image_idx, indices in enumerate(valid_indices):
            covs = output_array["output_cov/Sigmoid"][image_idx, :, :, :]
            bboxes = abs_bbox[image_idx, :, :, :]
            imagewise_boxes = []
            for class_idx in range(len(self.classes)):
                clustered_boxes = []
                cw_config = self.pproc_config.classwise_clustering_config[
                    self.classes[class_idx]
                ]
                classwise_covs = covs[class_idx, :, :].flatten()
                classwise_covs = classwise_covs[indices[class_idx]]
                if classwise_covs.size == 0:
                    continue
                classwise_bboxes = bboxes[4 * class_idx : 4 * class_idx + 4, :, :]
                classwise_bboxes = classwise_bboxes.reshape(
                    classwise_bboxes.shape[:1] + (-1,)
                ).T[indices[class_idx]]
                pairwise_dist = 1.0 * (1.0 - iou_vectorized(classwise_bboxes))
                labeling = self.dbscan_elements[self.classes[class_idx]].fit_predict(
                    X=pairwise_dist, sample_weight=classwise_covs
                )
                labels = np.unique(labeling[labeling >= 0])
                for label in labels:
                    w = classwise_covs[labeling == label]
                    aggregated_w = np.sum(w)
                    w_norm = w / aggregated_w
                    n = len(w)
                    w_max = np.max(w)
                    w_min = np.min(w)
                    b = classwise_bboxes[labeling == label]
                    mean_bbox = np.sum((b.T * w_norm).T, axis=0)

                    # Compute coefficient of variation of the box coords
                    mean_box_w = mean_bbox[2] - mean_bbox[0]
                    mean_box_h = mean_bbox[3] - mean_bbox[1]
                    bbox_area = mean_box_w * mean_box_h
                    valid_box = (
                        aggregated_w
                        > cw_config.dbscan_config.dbscan_confidence_threshold
                        and mean_box_h > cw_config.minimum_bounding_box_height
                    )
                    if valid_box:
                        batchwise_proba.append(aggregated_w)
                        clustered_boxes.append(mean_bbox)
                    else:
                        continue
                imagewise_boxes.extend(clustered_boxes)
            batchwise_boxes.append(imagewise_boxes)
        return batchwise_boxes, batchwise_proba
