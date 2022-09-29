import json
import logging
import os

import numpy as np
import triton_python_backend_utils as pb_utils
import tritonclient.grpc.model_config_pb2 as mc
from postprocessing.facenet_postprocessor import DetectNetPostprocessor
from postprocessing.types.frame import Frame

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

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "true_boxes"
        )

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        output1_config = pb_utils.get_output_config_by_name(
            self.model_config, "true_proba"
        )

        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

        self.data_format = mc.ModelInput.FORMAT_NCHW
        self.classes = ["face"]
        self.postprocessing_config = (
            "{}/postprocessing/clustering_config_facenet.prototxt".format(
                os.path.realpath(os.path.dirname(__file__))
            )
        )
        self.target_shape = (3, 416, 736)
        self.postprocessor = DetectNetPostprocessor(
            data_format=self.data_format,
            classes=self.classes,
            postprocessing_config=self.postprocessing_config,
            target_shape=self.target_shape,
        )

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
        output1_dtype = self.output1_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get inputs

            batchwise_boxes, batchwise_proba = self.postprocessor.apply(request, idx)

            out_tensor_0 = pb_utils.Tensor(
                "true_boxes", np.array(batchwise_boxes).astype(output0_dtype)
            )

            out_tensor_1 = pb_utils.Tensor(
                "true_proba", np.array(batchwise_proba).astype(output1_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
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
        logger.info("Postprocess cleaning up...")
