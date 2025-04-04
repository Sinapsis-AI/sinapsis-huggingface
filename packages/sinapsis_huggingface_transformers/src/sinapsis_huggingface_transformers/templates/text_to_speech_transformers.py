# -*- coding: utf-8 -*-

import numpy as np
import torch
from sinapsis_core.data_containers.data_packet import AudioPacket, DataContainer, TextPacket
from sinapsis_core.template_base import TemplateAttributeType

from sinapsis_huggingface_transformers.helpers import sentences_to_n_words, split_text_into_sentences
from sinapsis_huggingface_transformers.templates.base_transformers import (
    TransformersBase,
    TransformersBaseAttributes,
)


class TextToSpeechAttributes(TransformersBaseAttributes):
    """Attributes for the TextToSpeech template.

    Attributes:
        use_embeddings (bool): Whether to use speaker embeddings during audio generation.
            Defaults to `False`. If set to `True`, embeddings must be provided for each
            text packet.
        sample_rate (int | None): The sample rate (in Hz) for the generated audio.
            If not provided, the sample rate will be determined dynamically from the model.
        n_words (int | None): The number of words per sentence when splitting the
            input text into smaller chunks. If not specified, sentences will be split
            based on punctuation.
    """

    use_embeddings: bool = False
    sample_rate: int | None = None
    n_words: int | None = None


class TextToSpeechTransformers(TransformersBase):
    """
    The template generates an audio from a prompt that is passed
    through the text packet in the DataContainer.
    It uses the transformers architecture and a HuggingFace model to
    produce the audio. Finally, it sends the audio through the DataContainer

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: TextToSpeechTransformers
      class_name: TextToSpeechTransformers
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/model'
        model_cache_dir: /path/to/cache/dir
        device: 'cuda'
        torch_dtype: float16
        use_embeddings: false
        sample_rate: 16000
        n_words: 10

    """

    AttributesBaseModel = TextToSpeechAttributes

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.task = "text-to-speech"
        self.setup_pipeline()
        self.sample_rate = self._get_sample_rate()

    def _get_sample_rate(self) -> int:
        """Retrieve the sample rate for the generated audio.

        Returns:
            int: The sample rate (in Hz) for the audio.
        """
        if self.attributes.sample_rate:
            self.logger.info(f"Using provided sample rate: {self.attributes.sample_rate} Hz.")
            return self.attributes.sample_rate

        forward_params = (
            {"speaker_embeddings": torch.rand((1, 512), dtype=self._TORCH_DTYPE.get(self.attributes.torch_dtype))}
            if self.attributes.use_embeddings
            else {}
        )
        output = self.pipeline("Fetching sampling rate.", forward_params=forward_params)
        sample_rate = output.get("sampling_rate", 16000)

        return sample_rate

    def _split_text(self, text: str) -> list[str]:
        """Split the input text into smaller chunks for audio generation.

        Args:
            text (str): The input text to be split.

        Returns:
            list[str]: A list of text chunks.
        """
        if self.attributes.n_words:
            self.logger.info("Splitting the input text into sentences according to the number of words ...")
            return sentences_to_n_words(text, self.attributes.n_words)
        self.logger.info("Splitting input text into chunks by punctuation...")
        return split_text_into_sentences(text)

    def _get_audio_packet(self, text_packet: TextPacket) -> AudioPacket | None:
        """Convert input text into an audio packet using the pipeline.

        Args:
            text_packet (TextPacket): The text packet containing input text and optional embedding.

        Returns:
            AudioPacket | None: An audio packet containing the generated audio. Returns None if Audio could not be
                generated.
        """
        sentences = self._split_text(text_packet.content)
        self.logger.info(f"Number of sentences to be converted to audio={len(sentences)}")
        total_audio = []
        forward_params = (
            {"speaker_embeddings": torch.tensor(text_packet.embedding).unsqueeze(0)}
            if self.attributes.use_embeddings and text_packet.embedding
            else {}
        )
        for chunk in sentences:
            output = self.pipeline(chunk, forward_params=forward_params, **self.attributes.inference_kwargs)
            total_audio.append(output["audio"][0] if output["audio"].ndim == 2 else output["audio"])
        if total_audio:
            total_audio = np.concatenate(total_audio)
            audio_packet = AudioPacket(
                content=total_audio,
                sample_rate=self.sample_rate,
            )
            return audio_packet

        self.logger.warning("Audio packet could not be generated.")
        return None

    def transformation_method(self, container: DataContainer) -> DataContainer:
        """Convert input text or container texts into audio and update the container.

        Args:
            container (DataContainer): The input container holding text data.

        Returns:
            DataContainer: The updated container with audio packets.
        """
        audio_packets: list[AudioPacket] = []

        for text_packet in container.texts:
            audio_packet = self._get_audio_packet(text_packet)
            if audio_packet is not None:
                audio_packets.append(audio_packet)
                container.audios = audio_packets
        return container
