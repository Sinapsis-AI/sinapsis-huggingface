# -*- coding: utf-8 -*-

from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base.base_models import OutputTypes

from sinapsis_huggingface_transformers.helpers.tags import Tags
from sinapsis_huggingface_transformers.templates.base_transformers import (
    BaseInferenceKwargs,
    TransformersBase,
    TransformersBaseAttributes,
)

ImageToTextTransformersUIProperties = TransformersBase.UIProperties
ImageToTextTransformersUIProperties.output_type = OutputTypes.TEXT
ImageToTextTransformersUIProperties.tags.extend([Tags.IMAGE, Tags.TEXT, Tags.IMAGE_TO_TEXT])


class ImageToTextInferenceKwargs(BaseInferenceKwargs):
    """Specific keyword arguments for the image-to-text pipeline.

    Attributes:
        max_new_tokens (int | None): The maximum number of tokens to generate in the description.
        timeout (float | None): The maximum time in seconds to wait for fetching images from the web.
    """

    max_new_tokens: int | None = None
    timeout: float | None = None


class ImageToTextTransformersAttributes(TransformersBaseAttributes):
    """Defines the complete set of attributes for the ImageToTextTransformers template.

    Inherits general transformer settings from TransformersBaseAttributes.

    Attributes:
        inference_kwargs (ImageToTextInferenceKwargs): Task-specific parameters for the image-to-text pipeline,
            such as `max_new_tokens`.
    """

    model_path: Literal["microsoft/trocr-base-handwritten"] = "microsoft/trocr-base-handwritten"
    inference_kwargs: ImageToTextInferenceKwargs = Field(default_factory=ImageToTextInferenceKwargs)


class ImageToTextTransformers(TransformersBase):
    """ImageToTextTransformers template to generate text from an image.

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

    AttributesBaseModel = ImageToTextTransformersAttributes
    GENERATED_TEXT_KEY = "generated_text"
    UIProperties = ImageToTextTransformersUIProperties

    def initialize(self) -> None:
        """Initializes the template's common state for creation or reset.

        This method is called by both `__init__` and `reset_state` to ensure
        a consistent state.
        """
        super().initialize()
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
            results = self.pipeline(image, **self.attributes.inference_kwargs.model_dump(exclude_none=True))
            if results:
                text_description = results[0].get(self.GENERATED_TEXT_KEY)
                if text_description:
                    container.texts.append(TextPacket(content=text_description))
        return container
