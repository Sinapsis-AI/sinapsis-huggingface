# -*- coding: utf-8 -*-

from pydantic import Field
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base.base_models import OutputTypes

from sinapsis_huggingface_transformers.helpers.tags import Tags
from sinapsis_huggingface_transformers.templates.base_transformers import (
    BaseInferenceKwargs,
    TransformersBase,
    TransformersBaseAttributes,
)

TranslationTransformersUIProperties = TransformersBase.UIProperties
TranslationTransformersUIProperties.output_type = OutputTypes.TEXT
TranslationTransformersUIProperties.tags.extend([Tags.LANGUAGE, Tags.TRANSLATION])


class TranslationInferenceKwargs(BaseInferenceKwargs):
    """Specific keyword arguments for the translation pipeline.

    Attributes:
        return_text (bool | None): Whether or not to include the decoded texts in the outputs.
        return_tensors (bool | None): Whether or not to include the tensors of predictions.
        clean_up_tokenization_spaces (bool | None): Whether or not to clean up the potential extra spaces.
    """

    return_text: bool | None = True
    return_tensors: bool | None = False
    clean_up_tokenization_spaces: bool | None = False


class TranslationTransformersAttributes(TransformersBaseAttributes):
    """Attributes for the transformers pipeline translation task.

    Attributes:
        inference_kwargs: Task-specific parameters for the transaltion pipeline,
            such as `clean_up_tokenization_spaces`.
        source_language (str): The language code of the source language (e.g., "en" for English).
        target_language (str): The language code of the target language (e.g., "fr" for French).
    """

    inference_kwargs: TranslationInferenceKwargs = Field(default_factory=TranslationInferenceKwargs)
    source_language: str
    target_language: str


class TranslationTransformers(TransformersBase):
    """Template for text translation using a Hugging Face Transformers pipeline.

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
    UIProperties = TranslationTransformersUIProperties

    def initialize(self) -> None:
        """Initializes the template's common state for creation or reset.

        This method is called by both `__init__` and `reset_state` to ensure
        a consistent state.
        """
        super().initialize()
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
            results = self.pipeline(
                text_packet.content, **self.attributes.inference_kwargs.model_dump(exclude_none=True)
            )
            if results:
                translated_text = results[0].get(self.TRANSLATION_TEXT_KEY)
                if translated_text:
                    text_packet.content = translated_text
        return container
