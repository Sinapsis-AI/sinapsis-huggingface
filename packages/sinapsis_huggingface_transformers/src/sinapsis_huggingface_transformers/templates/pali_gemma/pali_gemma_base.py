# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Any, ClassVar, Literal

import torch
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import (
    Template,
    TemplateAttributes,
    TemplateAttributeType,
)
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


class PaliGemmaBaseAttributes(TemplateAttributes):
    """Base attributes for PaliGemma models.

    Attributes:
        model_path (str): Path to the pretrained PaliGemma model. Can be either:
            - A Hugging Face model identifier (e.g. 'facebook/pali-gemma-7b')
            - A local directory path containing the model files
        processor_path (str): Path to the model processor/tokenizer. Can be either:
            - A Hugging Face model identifier
            - A local directory path containing the processor files
        model_cache_dir (str): Directory for caching model files when downloading from Hugging Face.
        device (Literal["cuda", "cpu"]): Device to run the model on. Defaults to cpu.
        max_new_tokens (int): Maximum number of tokens to generate. Defaults to 200.
        torch_dtype (Literal["float16", "float32"]): Model precision type. Defaults to float16.
    """

    model_path: str
    processor_path: str
    model_cache_dir: str = str(SINAPSIS_CACHE_DIR)
    device: Literal["cuda", "cpu"] = "cpu"
    max_new_tokens: int = 200
    torch_dtype: Literal["float16", "float32"] = "float16"


class PaliGemmaBase(Template):
    """Base class for PaliGemma implementations."""

    AttributesBaseModel = PaliGemmaBaseAttributes
    CATEGORY = "Transformers"
    _TORCH_DTYPE: ClassVar[dict[str, Any]] = {"float16": torch.float16, "float32": torch.float32}

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.model = self._setup_model()
        self.processor = self._setup_processor()

    def _setup_model(
        self,
    ) -> PaliGemmaForConditionalGeneration:
        """Initialize model  with proper device placement and precision settings.

        Handles the loading of model components, configuring
        it according to the specified device and precision requirements.

        Returns:
            PaliGemmaForConditionalGeneration: Initialized and configured model.
        """

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.attributes.model_path,
            cache_dir=self.attributes.model_cache_dir,
            torch_dtype=self._TORCH_DTYPE.get(self.attributes.torch_dtype),
        ).to(self.attributes.device)

        return model

    def _setup_processor(self) -> AutoProcessor:
        """Initialize processor with proper device placement and precision settings.

        Handles the loading of processor components, configuring
        it according to the specified cache and precision requirements.

        Returns:
            AutoProcessor: Initialized and configured processor.
        """
        processor = AutoProcessor.from_pretrained(
            self.attributes.processor_path,
            cache_dir=self.attributes.model_cache_dir,
            torch_dtype=self._TORCH_DTYPE.get(self.attributes.torch_dtype),
        )
        return processor

    @abstractmethod
    def execute(self, container: DataContainer) -> DataContainer:
        """Execute method to be implemented by child classes.

        Args:
            container (DataContainer): The input data container to be processed.

        Returns:
            DataContainer: The processed container with model outputs.
        """
