# -*- coding: utf-8 -*-

import numpy as np
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base.template import TemplateAttributeType

from sinapsis_huggingface_transformers.templates.base_transformers import TransformersBase


class SpeechToTextTransformers(TransformersBase):
    """
    Template to perform speech-to-text actions
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

    TEXT_KEY = "text"

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
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
            if self.instance_name not in audio_packet.modified_by_templates:
                audio = audio_packet.content
                audio = audio.astype(np.float32)
                transcribed_text = self.pipeline(audio, **self.attributes.inference_kwargs)[self.TEXT_KEY]
                transcribed_text_textpacket = TextPacket(
                    content=transcribed_text,
                    source=audio_packet.source,
                )
                self.logger.info(f"Speech-to-text transcription: {transcribed_text}")
                container.texts.append(transcribed_text_textpacket)
        return container
