# -*- coding: utf-8 -*-

from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import (
    Template,
    TemplateAttributes,
    TemplateAttributeType,
)


class HuggingFaceEmbeddingExtractorAttributes(TemplateAttributes):
    """
    Attributes for HuggingFace embedding extraction.

        model_name (str): Name of the HuggingFace model for generating embeddings.
    """

    model_name: str


class HuggingFaceEmbeddingExtractor(Template):
    """
    The template includes functionality to generate text embeddings using
    HuggingFace models, and adding those embeddings to the annotations field
    of the TextPacket

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: HuggingFaceEmbeddingExtractor
      class_name: HuggingFaceEmbeddingExtractor
      template_input: InputTemplate
      attributes:
        model_name: 'embedding_model'

    """

    AttributesBaseModel = HuggingFaceEmbeddingExtractorAttributes
    CATEGORY = "Embeddings"

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.embed_model = HuggingFaceEmbedding(self.attributes.model_name)

    def generate_embedding(self, nodes: list[TextNode]) -> None:
        """Generates embeddings for a list of text nodes.

        Args:
            nodes (list[TextNode]): Nodes containing text to embed.
        """
        for node in nodes:
            node_embedding = self.embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
            node.embedding = node_embedding

    def generate_query_embedding(self, query: str) -> Embedding:
        """Generates an embedding for a query string.

        Args:
            query (str): Input text query.

        Returns:
            Embedding: Generated embedding.
        """
        query_embedding: Embedding = self.embed_model.get_query_embedding(query)
        return query_embedding

    def get_embeddings(self, container: DataContainer) -> DataContainer:
        """Computes embeddings for text packets in the container.

        Args:
            container (DataContainer): Container holding text data.

        Returns:
            DataContainer: Updated container with computed embeddings.
        """
        for packet in container.texts:
            embeddings = self.generate_query_embedding(packet.content)
            packet.embedding = embeddings
        return container

    def execute(self, container: DataContainer) -> DataContainer:
        """Processes the container to extract embeddings.

        Args:
            container (DataContainer): Container with text data.

        Returns:
            DataContainer: Updated container with extracted embeddings.
        """

        if not container.texts:
            self.logger.debug("No content to extract embedding from.")
            return container

        container = self.get_embeddings(container)
        return container
