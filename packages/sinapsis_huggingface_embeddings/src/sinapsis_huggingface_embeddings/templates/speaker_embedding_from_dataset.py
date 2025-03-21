# -*- coding: utf-8 -*-

from typing import Literal

from datasets import load_dataset
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import (
    Template,
    TemplateAttributes,
)
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR


class SpeakerEmbeddingFromDatasetAttributes(TemplateAttributes):
    """Attributes for the SpeakerEmbeddingFromDataset template.

    Attributes:
        dataset_path (str): Path or name of the Hugging Face dataset containing speaker embeddings.
            For example, `"Matthijs/cmu-arctic-xvectors"`.
        data_cache_dir (str): Directory to cache the downloaded dataset. Defaults to the value of
            the `SINAPSIS_CACHE_DIR` environment variable.
        split (str): Dataset split to use (e.g., "train", "validation", or "test").
            Defaults to `"validation"`.
        sample_idx (int): Index of the dataset sample to extract the embedding from.
        xvector_key (str): Key in the dataset sample that stores the xvector. Defaults to `"xvector"`.
        target_packet (Literal["texts", "audios"]): Type of packet in the `DataContainer` to which
            the embedding will be attached. Must be either `"texts"` or `"audios"`.
    """

    dataset_path: str
    data_cache_dir: str = str(SINAPSIS_CACHE_DIR)
    split: str = "validation"
    sample_idx: int
    xvector_key: str = "xvector"
    target_packet: Literal["texts", "audios"]


class SpeakerEmbeddingFromDataset(Template):
    """
    Template to retrieve and attach speaker embeddings from a Hugging Face dataset.
    This template extracts a specified embedding (e.g., xvector) from a dataset and attaches
    it to the `embedding` attribute of each `TextPacket` in a `DataContainer`.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: SpeakerEmbeddingFromDataset
      class_name: SpeakerEmbeddingFromDataset
      template_input: InputTemplate
      attributes:
        dataset_path: '/path/to/hugging/face/dataset'
        data_cache_dir: /path/to/cache/dir
        split: validation
        sample_idx: '1'
        xvector_key: xvector
        target_packet: 'audios'



    """

    AttributesBaseModel = SpeakerEmbeddingFromDatasetAttributes
    CATEGORY = "Embeddings"

    def execute(self, container: DataContainer) -> DataContainer:
        """Retrieve and attach speaker embeddings to specified packets in a DataContainer.

        Args:
            container (DataContainer): The container holding the packets to which the embedding will be
                attached.

        Returns:
            DataContainer: The updated container with embeddings attached to the `embedding`
                attribute of the specified packet type.
        """
        packets = getattr(container, self.attributes.target_packet)
        embeddings_dataset = load_dataset(
            self.attributes.dataset_path,
            split=self.attributes.split,
            cache_dir=self.attributes.data_cache_dir,
        )
        speaker_embedding = embeddings_dataset[self.attributes.sample_idx][self.attributes.xvector_key]
        self.logger.info(
            f"Attaching embedding from index {self.attributes.sample_idx} to "
            f"{len(packets)} {self.attributes.target_packet} packets."
        )
        for packet in packets:
            packet.embedding = speaker_embedding

        return container
