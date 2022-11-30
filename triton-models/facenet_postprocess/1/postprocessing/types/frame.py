import logging
import os

import numpy as np
import tritonclient.grpc.model_config_pb2 as mc
from postprocessing.preprocess_input import preprocess_input

logger = logging.getLogger(__name__)


class Frame(object):
    """Data structure to contain an image."""

    def __init__(self, image_path, data_format, dtype, target_shape):
        """Instantiate a frame object."""
        self._image_path = image_path
        if data_format not in [
                mc.ModelInput.FORMAT_NCHW,
                mc.ModelInput.FORMAT_NHWC]:
            raise NotImplementedError(
                "Data format not in the supported data format: {}".format(data_format))
        self.data_format = data_format
        self.height = None
        self.width = None
        self.dtype = dtype
        assert (
            len(target_shape) == 3
        ), "3 dimensions are required for input definitions. Got {}".format(
            len(target_shape)
        )
        if self.data_format == mc.ModelInput.FORMAT_NCHW:
            self.c, self.h, self.w = target_shape
        else:
            self.h, self.w, self.c = target_shape
        assert self.c in [
            3], "Number of channels should be 3. Got {}".format(self.c)
        self.target_shape = target_shape

        self.model_img_mode = "RGB" if self.c == 3 else "L"
        self.keep_aspect_ratio = True
        self.img_mean = None
