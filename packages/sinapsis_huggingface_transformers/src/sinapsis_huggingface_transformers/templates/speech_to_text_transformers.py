# -*- coding: utf-8 -*-
from typing import Literal

import numpy as np
from pydantic import Field
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base.base_models import OutputTypes

from sinapsis_huggingface_transformers.helpers.tags import Tags
from sinapsis_huggingface_transformers.templates.base_transformers import (
    BaseInferenceKwargs,
    TransformersBase,
    TransformersBaseAttributes,
)

SpeechToTextTransformersUIProperties = TransformersBase.UIProperties
SpeechToTextTransformersUIProperties.output_type = OutputTypes.TEXT
SpeechToTextTransformersUIProperties.tags.extend(
    [Tags.SPEECH, Tags.SPEECH_TO_TEXT, Tags.AUDIO, Tags.SPEECH_RECOGNITION, Tags.TEXT]
)


class SpeechToTextInferenceKwargs(BaseInferenceKwargs):
    """Specific keyword arguments for the automatic-speech-recognition pipeline.

    Attributes:
        return_timestamps (Literal["char", "word"] | bool | None ): If set, controls the granularity of
            timestamps returned with the transcribed text. Can be "char", "word", or True for segments.
    """

    return_timestamps: Literal["char", "word"] | bool | None = None


class SpeechToTextTransformersAttributes(TransformersBaseAttributes):
    """Defines the set of attributes for the SpeechToTextTransformers template.

    Inherits general transformer settings from TransformersBaseAttributes.

    Attributes:
        inference_kwargs (SpeechToTextInferenceKwargs): Task-specific parameters for the speech-to-text pipeline,
            such as `return_timestamps`.
    """

    model_path: Literal["openai/whisper-small"] = "openai/whisper-small"
    inference_kwargs: SpeechToTextInferenceKwargs = Field(default_factory=SpeechToTextInferenceKwargs)


class SpeechToTextTransformers(TransformersBase):
    """Template to perform speech-to-text actions
    using the HuggingFace module through the 'transformers' architecture.

    The template takes an Audio from the DataContainer and uses a speech-recognition
    model to transcribe the audio. Finally, it returns the text in the DataContainer

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: SpeechToTextTransformers
      class_name: SpeechToTextTransformers
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/model'
        model_cache_dir: /path/to/cache/dir
        device: 'cuda'
        torch_dtype: float16

    """

    AttributesBaseModel = SpeechToTextTransformersAttributes
    TEXT_KEY = "text"
    UIProperties = SpeechToTextTransformersUIProperties

    def initialize(self) -> None:
        """Initializes the template's common state for creation or reset.

        This method is called by both `__init__` and `reset_state` to ensure
        a consistent state.
        """
        super().initialize()
        self.task = "automatic-speech-recognition"
        self.setup_pipeline()

    def transformation_method(self, container: DataContainer) -> DataContainer:
        """Speech recognition (speech-to-text) using a Transformers Pipeline.

        Args:
            container (DataContainer): DataContainer including the audio to be
            transcribed.
        Returns:
            DataContainer: DataContainer including the transcribed audio.
        """
        for audio_packet in container.audios:
            audio = audio_packet.content
            audio = audio.astype(np.float32)
            results = self.pipeline(audio, **self.attributes.inference_kwargs.model_dump(exclude_none=True))
            if results:
                transcribed_text = results.get(self.TEXT_KEY)
                if transcribed_text:
                    self.logger.info(f"Speech-to-text transcription: {transcribed_text}")
                    container.texts.append(
                        TextPacket(
                            content=transcribed_text,
                            source=audio_packet.source,
                        )
                    )
        return container
