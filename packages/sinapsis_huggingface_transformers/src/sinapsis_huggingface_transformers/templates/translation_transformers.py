# -*- coding: utf-8 -*-


from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import TemplateAttributeType

from sinapsis_huggingface_transformers.templates.base_transformers import (
    TransformersBase,
    TransformersBaseAttributes,
)


class TranslationTransformersAttributes(TransformersBaseAttributes):
    """Attributes for the transformers pipeline translation task.

    Attributes:
        source_language (str): The language code of the source language (e.g., "en" for English).
        target_language (str): The language code of the target language (e.g., "fr" for French).
    """

    source_language: str
    target_language: str


class TranslationTransformers(TransformersBase):
    """
    Template for text translation using a Hugging Face Transformers pipeline.

    This class provides a reusable framework for translating text from one language
    to another using a pre-trained Hugging Face model. The source and target languages
    must be specified through the attributes.
    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: TranslationTransformers
      class_name: TranslationTransformers
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/model'
        model_cache_dir: /path/to/cache/dir
        device: 'cuda'
        torch_dtype: float16
        source_language: en
        target_language: fr

    """

    AttributesBaseModel = TranslationTransformersAttributes
    TRANSLATION_TEXT_KEY = "translation_text"

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.task = f"translation_{self.attributes.source_language}_to_{self.attributes.target_language}"
        self.setup_pipeline()

    def transformation_method(self, container: DataContainer) -> DataContainer:
        """Translate text using a Transformers Pipeline.

        Args:
            container (DataContainer): DataContainer including the text to be
            translated.

        Returns:
            DataContainer: DataContainer including the translated text.
        """
        for text_packet in container.texts:
            translated_text = self.pipeline(text_packet.content, **self.attributes.inference_kwargs)[0][
                self.TRANSLATION_TEXT_KEY
            ]
            text_packet.content = translated_text
        return container
