# -*- coding: utf-8 -*-

from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import TemplateAttributeType

from sinapsis_huggingface_transformers.templates.base_transformers import TransformersBase


class SummarizationTransformers(TransformersBase):
    """
    Template for text summarization using a Hugging Face Transformers pipeline.

    This class provides a reusable framework for summarizing text using a pre-trained
    Hugging Face model. The `max_length` and `min_length` attributes control the length
    of the generated summaries.

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

    """

    SUMMARY_TEXT_KEY = "summary_text"

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
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
            original_text = text_packet.content
            summarized_text = self.pipeline(
                original_text,
                max_length=self.attributes.max_length,
                min_length=self.attributes.min_length,
            )[0][self.SUMMARY_TEXT_KEY]

            text_packet.content = summarized_text
        return container
