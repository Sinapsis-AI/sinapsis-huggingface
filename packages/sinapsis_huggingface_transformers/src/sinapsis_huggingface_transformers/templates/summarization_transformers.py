# -*- coding: utf-8 -*-

from typing import Literal

from pydantic import Field
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base.base_models import OutputTypes

from sinapsis_huggingface_transformers.helpers.tags import Tags
from sinapsis_huggingface_transformers.templates.base_transformers import (
    BaseInferenceKwargs,
    TransformersBase,
    TransformersBaseAttributes,
)

SummarizationTransformersUIProperties = TransformersBase.UIProperties
SummarizationTransformersUIProperties.output_type = OutputTypes.TEXT
SummarizationTransformersUIProperties.tags.extend([Tags.SUMMARIZATION, Tags.TEXT])


class SummarizationInferenceKwargs(BaseInferenceKwargs):
    """Specific keyword arguments for the summarization pipeline.

    Attributes:
        return_text (bool | None): Whether or not to include the decoded texts in the outputs.
        return_tensors (bool | None): Whether or not to include the tensors of predictions.
        clean_up_tokenization_spaces (bool | None): Whether or not to clean up the potential extra spaces.
    """

    return_text: bool | None = True
    return_tensors: bool | None = False
    clean_up_tokenization_spaces: bool | None = False


class SummarizationTransformersAttributes(TransformersBaseAttributes):
    """Defines the complete set of attributes for the SummarizationTransformers template.

    Inherits general transformer settings from TransformersBaseAttributes.

    Attributes:
        inference_kwargs: Task-specific parameters for the summarization pipeline,
            such as `clean_up_tokenization_spaces`.
    """

    model_path: Literal[
        "facebook/bart-large-cnn", "google/pegasus-xsum", "Falconsai/text_summarization", "facebook/bart-large-xsum"
    ] = "facebook/bart-large-cnn"

    inference_kwargs: SummarizationInferenceKwargs = Field(default_factory=SummarizationInferenceKwargs)


class SummarizationTransformers(TransformersBase):
    """Template for text summarization using a Hugging Face Transformers pipeline.

    This class provides a reusable framework for summarizing text using a pre-trained
    Hugging Face model.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: SummarizationTransformers
      class_name: SummarizationTransformers
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/model'
        model_cache_dir: /path/to/cache/dir
        device: 'cuda'
        torch_dtype: float16
        inference_kwargs:
            min_length: 5
            max_length: 20

    """

    AttributesBaseModel = SummarizationTransformersAttributes
    SUMMARY_TEXT_KEY = "summary_text"
    UIProperties = SummarizationTransformersUIProperties

    def initialize(self) -> None:
        """Initializes the template's common state for creation or reset.

        This method is called by both `__init__` and `reset_state` to ensure
        a consistent state.
        """
        super().initialize()
        self.task = "summarization"
        self.setup_pipeline()

    def transformation_method(self, container: DataContainer) -> DataContainer:
        """Summarize text using a Transformers Pipeline.

        Args:
            container (DataContainer): DataContainer including the text to be
            summarized.

        Returns:
            DataContainer: DataContainer including the summarized text.
        """
        for text_packet in container.texts:
            results = self.pipeline(
                text_packet.content, **self.attributes.inference_kwargs.model_dump(exclude_none=True)
            )
            if results:
                summarized_text = results[0].get(self.SUMMARY_TEXT_KEY)
                if summarized_text:
                    text_packet.content = summarized_text
        return container
