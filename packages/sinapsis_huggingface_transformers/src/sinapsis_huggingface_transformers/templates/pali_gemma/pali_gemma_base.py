# -*- coding: utf-8 -*-
import gc
from abc import abstractmethod
from typing import Any, ClassVar, Literal

import torch
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR
from sinapsis_huggingface_transformers.helpers.tags import Tags
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

    model_path: Literal["google/paligemma2-3b-pt-224"] = "google/paligemma2-3b-pt-224"
    processor_path: str
    model_cache_dir: str = str(SINAPSIS_CACHE_DIR)
    device: Literal["cuda", "cpu"] = "cpu"
    max_new_tokens: int = 200
    torch_dtype: Literal["float16", "float32"] = "float16"


class PaliGemmaBase(Template):
    """Base class for PaliGemma implementations."""

    AttributesBaseModel = PaliGemmaBaseAttributes
    UIProperties = UIPropertiesMetadata(
        category="HuggingFace",
        output_type=OutputTypes.IMAGE,
        tags=[Tags.HUGGINGFACE, Tags.IMAGE, Tags.PALIGEMMA, Tags.MODELS],
    )
    _TORCH_DTYPE: ClassVar[dict[str, Any]] = {"float16": torch.float16, "float32": torch.float32}

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.initialize()

    def initialize(self) -> None:
        """Initializes the template's common state for creation or reset.

        This method is called by both `__init__` and `reset_state` to ensure
        a consistent state. Can be overriden by subclasses for specific behaviour.
        """
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

    def reset_state(self, template_name: str | None = None) -> None:
        """Releases the model and processor from memory and re-instantiates the template.

        Args:
            template_name (str | None, optional): The name of the template instance being reset. Defaults to None.
        """
        _ = template_name

        if hasattr(self, "model"):
            self.model.to("cpu")
            del self.model

        if hasattr(self, "processor"):
            del self.processor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.initialize()
        self.logger.info(f"Reset template instance `{self.instance_name}`")
