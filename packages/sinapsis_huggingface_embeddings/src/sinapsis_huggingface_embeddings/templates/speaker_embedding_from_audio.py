# -*- coding: utf-8 -*-

import io
from typing import Literal

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from sinapsis_core.data_containers.data_packet import AudioPacket, DataContainer
from sinapsis_core.template_base import Template, TemplateAttributes, TemplateAttributeType
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR
from speechbrain.inference.speaker import EncoderClassifier


class SpeakerEmbeddingFromAudioAttributes(TemplateAttributes):
    """Attributes for SpeakerEmbeddingFromAudio template.

    Attributes:
        model_path (str): HuggingFace source used to load the model.
            Defaults to "speechbrain/spkrec-xvect-voxceleb".
        data_cache_dir (str): Directory in which the artifacts of the loaded model will be stored.
            Defaults to `SINAPSIS_CACHE_DIR`.
        target_packet (Literal["texts", "audios"]): Type of packet in the `DataContainer` to which
            the embedding will be attached. Must be either `"texts"` or `"audios"`.
        normalize (bool): Whether to apply normalization to the generated embeddings.
            If `True`, the embeddings will be normalized using the model's internal
            normalization mechanism, ensuring compatibility for tasks like voice verification.
            If `False`, the raw embeddings will be used without additional normalization.
            Defaults to `False`.
    """

    model_path: str = "speechbrain/spkrec-xvect-voxceleb"
    data_cache_dir: str = str(SINAPSIS_CACHE_DIR)
    target_packet: Literal["texts", "audios"]
    normalize: bool = False


class SpeakerEmbeddingFromAudio(Template):
    """
    The template provides functionality to generate embeddings
    for an audio provided in the DataContainer, using Huggingface models

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: SpeakerEmbeddingFromAudio
      class_name: SpeakerEmbeddingFromAudio
      template_input: InputTemplate
      attributes:
        model_path: speechbrain/spkrec-xvect-voxceleb
        data_cache_dir: /path/to/cache/dir
        target_packet: 'audios'
        normalize: false


    """

    AttributesBaseModel = SpeakerEmbeddingFromAudioAttributes
    CATEGORY = "Embeddings"

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.classifier = self._get_encoder()

    def _get_encoder(self) -> EncoderClassifier:
        """Load the speaker embedding model.

        Returns:
            EncoderClassifier: A pre-trained model for speaker embedding extraction.
        """
        return EncoderClassifier.from_hparams(source=self.attributes.model_path, savedir=self.attributes.data_cache_dir)

    @staticmethod
    def _postprocess_speaker_embedding(speaker_embedding: torch.Tensor) -> list[float]:
        """Normalize and convert the speaker embedding tensor into a list of floats.

        Args:
            speaker_embedding (torch.Tensor): The raw embedding tensor.

        Returns:
            list[float]: A normalized and flattened embedding as a list of floats.
        """
        speaker_embedding = F.normalize(speaker_embedding, dim=2)
        speaker_embedding_list: list[float] = speaker_embedding.squeeze().tolist()
        return speaker_embedding_list

    @staticmethod
    def _process_audio(audio_packet: AudioPacket) -> torch.Tensor:
        """Convert audio input (bytes or NumPy array) into a PyTorch tensor with the required
        sample rate.

        Args:
            audio_packet (AudioPacket): Audio packet containing the audio content and metadata.

        Returns:
            torch.Tensor: The processed audio tensor, resampled to 16kHz if necessary.
        """
        if isinstance(audio_packet.content, bytes):
            audio, original_sample_rate = sf.read(io.BytesIO(audio_packet.content))
        else:
            audio = audio_packet.content
            original_sample_rate = audio_packet.sample_rate

        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        if original_sample_rate != 16000:
            audio_tensor = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=16000)(audio_tensor)

        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=-1)

        return audio_tensor

    def execute(self, container: DataContainer) -> DataContainer:
        """Extract speaker embeddings from audio and attach them to the target packets.

        Args:
            container (DataContainer): A container holding the input audio and target packets.

        Raises:
            ValueError: If no audio packets are found in the container.
            ValueError: If the number of audio packets does not match the number of target packets,
                unless there is exactly one audio packet.

        Returns:
            DataContainer: The updated container with speaker embeddings attached to the target
                packets.
        """
        if not container.audios:
            raise ValueError("No audio packets found in the container.")

        embeddings = [
            self._postprocess_speaker_embedding(
                self.classifier.encode_batch(self._process_audio(audio_packet), normalize=self.attributes.normalize)
            )
            for audio_packet in container.audios
        ]

        packets = getattr(container, self.attributes.target_packet)

        if len(container.audios) == 1:
            for packet in packets:
                packet.embedding = embeddings[0]
        elif len(container.audios) == len(packets):
            for packet, embedding in zip(packets, embeddings):
                packet.embedding = embedding
        else:
            raise ValueError(
                "Mismatch between the number of audio packets and target packets. "
                "Ensure either a single audio or matching numbers of audios and target packets."
            )

        return container
