import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def

SAVE_AS = "/models/fpenet_preprocess/1/model.dali"


class FPENetPipeline:
    """Grayscale face clips reshaped to 1 x 80 X 80"""

    def __init__(self):
        self.raw_image_tensor = fn.external_source(name="raw_image_data")
        self.bboxes = fn.external_source(name="true_boxes")
        self.shapes = fn.peek_image_shape(self.raw_image_tensor)

    def load_images(self):
        self.image_tensor = fn.decoders.image(
            self.raw_image_tensor,
            output_type=types.GRAY,
            device="mixed",
        )

    def slice_bbox(self):
        self.image_tensor = fn.slice(
            self.image_tensor,
            start=self.bboxes[:2],
            end=self.bboxes[2:],
            out_of_bounds_policy="trim_to_shape",
        )

    def resize_slice_bbox(self):
        self.image_tensor = fn.resize(
            self.image_tensor,
            resize_x=80,
            resize_y=80,
            interp_type=types.DALIInterpType.INTERP_LANCZOS3,
            dtype=types.FLOAT,
        )

    def transpose_images(self):
        self.image_tensor = fn.transpose(self.image_tensor, perm=[2, 0, 1])

    @pipeline_def(batch_size=32, num_threads=64)
    def fpenet_transform(self):
        self.load_images()
        self.slice_bbox()
        self.resize_slice_bbox()
        self.transpose_images()

        return self.image_tensor


fpenet_pipeline = FPENetPipeline()
_ = fpenet_pipeline.fpenet_transform().serialize(filename=SAVE_AS)
