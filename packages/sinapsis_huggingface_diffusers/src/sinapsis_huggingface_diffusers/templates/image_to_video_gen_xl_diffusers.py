# -*- coding: utf-8 -*-

from diffusers import I2VGenXLPipeline
from sinapsis_core.template_base.template import TemplateAttributeType

from sinapsis_huggingface_diffusers.templates.image_to_image_diffusers import (
    ImageToImageDiffusers,
)


class ImageToVideoGenXLDiffusers(ImageToImageDiffusers):
    """This class implements a specific template for image-to-video generation using Hugging Face's
    diffusers. The `ImageToVideoGenXLDiffusers` class inherits from the `ImageToImageDiffusers` template
    to define how to handle converting input images into videos with the I2VGen-XL pipeline.

    Parameters to be used inside generation_params can be seen at HuggingFace documentation:
    https://huggingface.co/docs/diffusers/api/pipelines/i2vgenxl

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: ImageToVideoGenXLDiffusers
      class_name: ImageToVideoGenXLDiffusers
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/diffusers/model'
        model_cache_dir: /path/to/cache/dir
        device: 'cuda'
        torch_dtype: float16
        enable_model_cpu_offload: false
        generation_params: {}
        overwrite_images: false

    """

    DEFAULT_NUM_FRAMES = 16

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.num_duplicates = self.attributes.generation_params.get("num_frames", self.DEFAULT_NUM_FRAMES)
        self.requires_pil = True
        self.output_attribute = "frames"

    @staticmethod
    def _pipeline_class() -> I2VGenXLPipeline:
        """Returns the I2VGenXLPipeline class to be used for the image-to-video generative task.

        This method specifies the pipeline class required for image-to-video tasks. It ensures the
        correct pipeline is used for generating videos from input images.

        Returns:
            I2VGenXLPipeline: The class reference for I2VGenXLPipeline.
        """
        return I2VGenXLPipeline
