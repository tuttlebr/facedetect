import io
import json
import logging
import os

import dlib
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()
logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = json.loads(args["model_config"])

        self.output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "face_descriptor"
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            self.output0_config["data_type"]
        )

        self.face_rec_model_path = (
            "{}/dlib/dlib_face_recognition_resnet_model_v1.dat".format(
                os.path.realpath(os.path.dirname(__file__))
            )
        )

        self.facerec = dlib.face_recognition_model_v1(self.face_rec_model_path)

    def get_descriptors(self, face_clip):
        try:
            descriptor = np.array(
                self.facerec.compute_face_descriptor(face_clip.squeeze())
            )
        except Exception as e:
            logger.info("There was an exception: {}".format(e))
            descriptor = np.zeros((128,))

        return descriptor

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype
        responses = []

        for request in requests:
            image_tensor = pb_utils.get_input_tensor_by_name(
                request, "face_clip"
            ).as_numpy()

            dlib_descriptors = self.get_descriptors(image_tensor)

            out_tensor_0 = pb_utils.Tensor(
                "face_descriptor", np.array(dlib_descriptors).astype(output0_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        logger.info("Face descriptor cleaning up...")
