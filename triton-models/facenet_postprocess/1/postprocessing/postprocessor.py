class Postprocessor(object):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, data_format):
        """Initialize a post processor class.
        Args:
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.
        """
        self.data_format = data_format
        self.initialized = True

    def apply(self, output_tensors, this_id):
        """Apply the post processor to the outputs."""
        raise NotImplementedError("Base class doesn't implement any post-processing")
