# -*- coding: utf-8 -*-

import gc
import random
from abc import abstractmethod
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import TemplateAttributes, TemplateAttributeType, UIPropertiesMetadata
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR
from transformers import AutoProcessor, pipeline
from transformers.pipelines import Pipeline

from sinapsis_huggingface_transformers.helpers.tags import Tags


class BaseInferenceKwargs(BaseModel):
    """A flexible container for keyword arguments passed during inference.

    Attributes:
        generate_kwargs (dict[str, Any] | None): A dictionary of advanced parameters passed directly to the
            model's `generate` method for fine-tuning the pipeline generation.
    """

    generate_kwargs: dict[str, Any] | None = None
    model_config = ConfigDict(extra="allow")


class PipelineKwargs(BaseModel):
    """A flexible container for keyword arguments used to create the pipeline.

    This model allows any extra parameters to be passed during pipeline instantiation.
    """

    device: Literal["cuda", "cpu"]
    torch_dtype: Literal["float16", "float32", "auto"] = "float16"
    model_config = ConfigDict(extra="allow")


class TransformersBaseAttributes(TemplateAttributes):
    """Attributes for configuring the TransformersPipelineTemplate.

    Attributes:
        model_path (str): Name or path of the model from Hugging Face (e.g.,
            `openai/whisper-small.en`).
        model_cache_dir (str): Directory to cache the model files.
        device (Literal["cuda", "cpu"]): Device to run the pipeline on, either "cuda" for GPU or
            "cpu".
        torch_dtype (Literal["float16", "float32"]): Data type for PyTorch tensors; "float16" for
            half precision and "float32" for full precision.
        seed (int | None): Random seed for reproducibility. If provided, this seed will ensure
            consistent results for pipelines that involve randomness. If not provided, a random seed
            will be generated internally.
        pipeline_kwargs (PipelineKwargs): Keyword arguments passed during the instantiation of the
            Hugging Face pipeline.
        inference_kwargs (BaseInferenceKwargs): Keyword arguments passed during the task execution or
            inference phase. These allow dynamic customization of the task, such as `max_length`
            and `min_length` for summarization, or `max_new_tokens` for image-to-text.
    """

    model_path: str
    model_cache_dir: str = str(SINAPSIS_CACHE_DIR)
    seed: int | None = None
    pipeline_kwargs: PipelineKwargs = Field(default_factory=PipelineKwargs)
    inference_kwargs: BaseInferenceKwargs = Field(default_factory=BaseInferenceKwargs)


class TransformersBase(Template):
    """Base class for implementing task-specific Hugging Face Transformers pipelines.

    This class provides a reusable framework for tasks such as speech recognition,
    image-to-text, translation, and others. Subclasses must define the specific
    transformation logic in the `transformation_method`.
    """

    AttributesBaseModel = TransformersBaseAttributes
    UIProperties = UIPropertiesMetadata(
        category="Transformers",
        tags=[Tags.HUGGINGFACE, Tags.TRANSFORMERS, Tags.MODELS],
    )

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self._TORCH_DTYPE = {"float16": torch.float16, "float32": torch.float32}
        self.task: str | None = None
        self.initialize()

    def initialize(self) -> None:
        """Initializes the template's common state for creation or reset.

        This method is called by both `__init__` and `reset_state` to ensure
        a consistent state. Can be overriden by subclasses for specific behaviour.
        """
        self._set_seed()

    def setup_pipeline(self) -> None:
        """Initialize and configure the HuggingFace Transformers processing pipeline.

        Raises:
            ValueError: If called before the task attribute is set. The task must be
                defined by the child class before pipeline initialization.
        """
        if self.task is None:
            raise ValueError("'task' must be assigned before pipeline setup")

        self.processor = self._initialize_processor()
        self.pipeline = self.initialize_pipeline()

    def _set_seed(self) -> None:
        """Set the random seed for reproducibility. If no seed is provided, a random one will
        be generated.
        """

        seed = self.attributes.seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        self.logger.info(f"Seed set for reproducibility: {seed}")

    def _initialize_processor(self) -> AutoProcessor:
        """Initialize and return the processor for the model.

        Returns:
            AutoProcessor: Processor instance loaded from the model.
        """
        return AutoProcessor.from_pretrained(
            self.attributes.model_path,
            cache_dir=self.attributes.model_cache_dir,
        )

    def initialize_pipeline(self, **kwargs: dict[str, Any]) -> Pipeline:
        """Initialize and return the Transformers pipeline for the specified task.

        Subclasses can override this method to provide additional task-specific
        arguments via `kwargs`.

        Returns:
            pipeline: Hugging Face Transformers pipeline initialized with the
                      provided model and configuration.
        """
        return pipeline(
            task=self.task,
            model=self.attributes.model_path,
            **self.attributes.pipeline_kwargs.model_dump(),
            **kwargs,
        )

    @abstractmethod
    def transformation_method(self, container: DataContainer) -> DataContainer:
        """Abstract method to transform the input data container.

        Subclasses must implement this method to define the task-specific logic
        for transforming the input data.

        Args:
            container (DataContainer): The input data container to be transformed.

        Returns:
            DataContainer: The transformed data container.
        """

    def execute(self, container: DataContainer) -> DataContainer:
        """Apply a transforms pipeline according to the task.

        Args:
            container (Optional[DataContainer], optional): input DataContainer. Defaults to None.

        Returns:
            DataContainer: output DataContainer.
        """
        transformed_data_container = self.transformation_method(container)
        return transformed_data_container

    def reset_state(self, template_name: str | None = None) -> None:
        """Releases the pipeline and processor from memory and re-instantiates the template.

        Args:
            template_name (str | None, optional): The name of the template instance being reset. Defaults to None.
        """
        _ = template_name

        if hasattr(self, "pipeline") and self.pipeline is not None:
            if self.pipeline.model is not None:
                self.pipeline.model.to("cpu")
            del self.pipeline

        if hasattr(self, "processor"):
            del self.processor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.initialize()
        self.logger.info(f"Reset template instance `{self.instance_name}`")
