# -*- coding: utf-8 -*-
from typing import Literal

from sinapsis_core.data_containers.annotations import TextAnnotations
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base.base_models import OutputTypes

from sinapsis_huggingface_transformers.helpers.tags import Tags
from sinapsis_huggingface_transformers.templates.base_transformers import (
    TransformersBase,
    TransformersBaseAttributes,
)

TextToSpeechTransformersUIProperties = TransformersBase.UIProperties
TextToSpeechTransformersUIProperties.output_type = OutputTypes.TEXT
if TextToSpeechTransformersUIProperties.tags is not None:
    TextToSpeechTransformersUIProperties.tags.extend([Tags.TEXT, Tags.SENTIMENT_ANALYSIS])


class SentimentAnalysisAttributes(TransformersBaseAttributes):
    """Attributes for the SentimentAnalysis template.

    Attributes:
        model_path (Literal[str]):
        Model from Huggingface hub that accepts sentiment-analysis or zero-short-classification
         in the pipeline.

         Options for zero-shot-classification:"MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                                              "facebook/bart-large-mnli",
                                              "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
                                              "joeddav/xlm-roberta-large-xnli"



         Options for sentiment-analysis: "finiteautomata/bertweet-base-sentiment-analysis",
                                         "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                                        "nlptown/bert-base-multilingual-uncased-sentiment",
                                        "tabularisai/multilingual-sentiment-analysis",
    """

    model_path: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    task: Literal["zero-shot-classification", "sentiment-analysis"] = "zero-shot-classification"


class SentimentAnalysisTransformers(TransformersBase):
    """This template collects all the content in the text packets and runs inference
    using the sentiment-analysis task within the pipeline. Depending on the model, the
    label can be as simple as positive-negative or more complex.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: SentimentAnalysisTransformers
      class_name: SentimentAnalysisTransformers
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/model'


    """

    AttributesBaseModel = SentimentAnalysisAttributes
    UIProperties = TextToSpeechTransformersUIProperties
    attributes: SentimentAnalysisAttributes

    def initialize(self) -> None:
        """Initializes the template's common state for creation or reset.

        This method is called by both `__init__` and `reset_state` to ensure
        a consistent state.
        """
        super().initialize()
        self.task = self.attributes.task
        self.setup_pipeline()

    def transformation_method(self, container: DataContainer) -> DataContainer:
        """Convert input text or container texts into a list of strings. Passes the list through the pipeline
        and returns the label and confidence score as text packet annotations.

        Args:
            container (DataContainer): The input container holding text data.

        Returns:
            DataContainer: The updated container with audio packets.
        """
        text_to_analise = [text_packet.content for text_packet in container.texts]

        sentiments = self.pipeline(text_to_analise, **self.attributes.inference_kwargs.model_dump(exclude_none=True))
        for i, text_packet in enumerate(container.texts):
            if self.attributes.task == "zero-shot-classification":
                annotations = [
                    TextAnnotations(label=label, confidence_score=score)
                    for label, score in zip(sentiments[i].get("labels"), sentiments[i].get("scores"))
                ]

            else:
                annotations = [
                    TextAnnotations(label_str=sentiments[i].get("label"), confidence_score=sentiments[i].get("score"))
                ]

            text_packet.annotations.extend(annotations)
        return container
