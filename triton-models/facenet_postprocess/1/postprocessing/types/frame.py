import logging
import os

import numpy as np
import tritonclient.grpc.model_config_pb2 as mc
from PIL import Image
from pillow_heif import register_heif_opener
from postprocessing.preprocess_input import preprocess_input

register_heif_opener()
logger = logging.getLogger(__name__)


class Frame(object):
    """Data structure to contain an image."""

    def __init__(self, image_path, data_format, dtype, target_shape):
        """Instantiate a frame object."""
        self._image_path = image_path
        if data_format not in [mc.ModelInput.FORMAT_NCHW, mc.ModelInput.FORMAT_NHWC]:
            raise NotImplementedError(
                "Data format not in the supported data format: {}".format(data_format)
            )
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
        assert self.c in [3], "Number of channels should be 3. Got {}".format(self.c)
        self.target_shape = target_shape

        self.model_img_mode = "RGB" if self.c == 3 else "L"
        self.keep_aspect_ratio = True
        self.img_mean = None

    def load_image(self):
        """Load the image defined."""
        if not os.path.exists(self._image_path):
            raise NotFoundError("Cannot find image at {}".format(self._image_path))
        image = Image.open(self._image_path)
        self.width, self.height = image.size

        if self.c == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        return image

    def as_numpy(self, image):
        """Return a numpy array."""
        image = image.resize((self.w, self.h), Image.ANTIALIAS)
        nparray = np.asarray(image).astype(self.dtype)
        if nparray.ndim == 2:
            nparray = nparray[:, :, np.newaxis]
        if self.data_format == mc.ModelInput.FORMAT_NCHW:
            nparray = np.transpose(nparray, (2, 0, 1))
        return nparray * (1 / 255.0)

    def _load_img(self):
        """load an image and returns the original image and a numpy array for model to consume.
        Args:
            img_path (str): path to an image
        Returns:
            img (PIL.Image): PIL image of original image.
            ratio (float): resize ratio of original image over processed image
            inference_input (array): numpy array for processed image
        """
        img = Image.open(self._image_path)
        orig_w, orig_h = img.size
        ratio = min(self.w / float(orig_w), self.h / float(orig_h))

        # do not change aspect ratio
        new_w = int(round(orig_w * ratio))
        new_h = int(round(orig_h * ratio))

        if self.keep_aspect_ratio:
            im = img.resize((new_w, new_h), Image.ANTIALIAS)
        else:
            im = img.resize((self.w, self.h), Image.ANTIALIAS)

        if (
            im.mode in ("RGBA", "LA")
            or (im.mode == "P" and "transparency" in im.info)
            and self.model_img_mode == "L"
        ):

            # Need to convert to RGBA if LA format due to a bug in PIL
            im = im.convert("RGBA")
            inf_img = Image.new("RGBA", (self.w, self.h))
            inf_img.paste(im, (0, 0))
            inf_img = inf_img.convert(self.model_img_mode)
        else:
            inf_img = Image.new(self.model_img_mode, (self.w, self.h))
            inf_img.paste(im, (0, 0))

        inf_img = np.array(inf_img).astype(np.float32)
        if self.model_img_mode == "L":
            inf_img = np.expand_dims(inf_img, axis=2)
            inference_input = inf_img.transpose(2, 0, 1) - 117.3786
        else:
            inference_input = preprocess_input(
                inf_img.transpose(2, 0, 1), img_mean=self.img_mean
            )

        return inference_input
