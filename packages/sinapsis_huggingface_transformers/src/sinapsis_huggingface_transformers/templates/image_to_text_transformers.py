# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base.template import TemplateAttributeType

from sinapsis_huggingface_transformers.templates.base_transformers import TransformersBase


class ImageToTextTransformers(TransformersBase):
    """
    ImageToTextTransformers template to generate text from an image.

    This template uses a Hugging Face Transformers pipeline to generate textual descriptions
    from input images.
    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: ImageToTextTransformers
      class_name: ImageToTextTransformers
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/model'
        model_cache_dir: /path/to/cache/dir
        device: 'cuda'
        torch_dtype: float16

    """

    GENERATED_TEXT_KEY = "generated_text"

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.task = "image-to-text"
        self.setup_pipeline()

    @staticmethod
    def _convert_to_pil(image_content: Image.Image | np.ndarray) -> Image.Image:
        """Convert image content to a PIL Image.

        This method checks the type of the image content and converts it to a PIL Image
        if it's a NumPy array. If the content is already a PIL Image, it returns it as-is.

        Args:
            image_content (Image.Image | np.ndarray): The input image content.

        Returns:
            Image.Image: The input image as a PIL Image.
        """
        if isinstance(image_content, Image.Image):
            return image_content
        return Image.fromarray(image_content)

    def transformation_method(self, container: DataContainer) -> DataContainer:
        """Generate text descriptions for images using the configured Transformers pipeline.

        Args:
            container (DataContainer): A DataContainer holding images to be described.

        Returns:
            DataContainer: The updated DataContainer with text descriptions added.
        """
        for image_packet in container.images:
            image = self._convert_to_pil(image_packet.content)
            text_description = self.pipeline(image, **self.attributes.inference_kwargs)[0][self.GENERATED_TEXT_KEY]
            text_packet = TextPacket(content=text_description)
            container.texts.append(text_packet)
        return container
