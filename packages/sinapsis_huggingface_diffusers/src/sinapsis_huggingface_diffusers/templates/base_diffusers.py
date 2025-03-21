# -*- coding: utf-8 -*-

import gc
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import torch
from diffusers import DiffusionPipeline
from pydantic import BaseModel, ConfigDict
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket
from sinapsis_core.template_base import (
    Template,
    TemplateAttributes,
    TemplateAttributeType,
)
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR


class BaseDiffusersAttributes(TemplateAttributes):
    """Configuration attributes for setting up a diffusion pipeline and generating images.

    Attributes:
        model_path (str): The path or identifier of the pre-trained model to load.
            - Can be a repository ID from Hugging Face's model hub (e.g.,
                "CompVis/stable-diffusion-v1-4").
            - Can also be a local directory path containing the model weights and configuration
                files.
        model_cache_dir (str): Directory to cache and save the models.
        device (Literal["cuda", "cpu"]): Device for computations, either "cpu" or "cuda".
        torch_dtype (Literal["float16", "float32"]): Data type for PyTorch tensors.
        enable_model_cpu_offload (bool): If True, enables CPU offloading to reduce GPU memory usage.
        generation_params (dict): Parameters for image generation (e.g., prompt, guidance_scale).
        seed (int | list[int] | None): Random seed(s) for reproducibility.
        overwrite_images (bool): Whether to overwrite the existing images in the container.
            Defaults to False.
    """

    model_path: str
    model_cache_dir: str = str(SINAPSIS_CACHE_DIR)
    device: Literal["cuda", "cpu"]
    torch_dtype: Literal["float16", "float32"] = "float16"
    enable_model_cpu_offload: bool = False
    generation_params: dict[str, Any]
    seed: int | list[int] | None = None
    overwrite_images: bool = False


class TorchTypes(BaseModel):
    """BaseModel to contain torch data types"""

    float16: torch.dtype = torch.float16
    float32: torch.dtype = torch.float32

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class BaseDiffusers(Template, ABC):
    """Base template for generative tasks using Hugging Face diffusers.

    This abstract class provides the foundational methods for setting up and executing a diffusion
    pipeline using pre-trained models. Specific tasks like text-to-image or inpainting should
    inherit from this class and implement `_pipeline_class` and `execute`.
    """

    AttributesBaseModel = BaseDiffusersAttributes
    CATEGORY = "Diffusers"
    TORCH_DTYPE: dict = TorchTypes().model_dump()

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)

        self.pipeline = self._make_pipeline()
        self.pipeline.set_progress_bar_config(disable=True)
        self.num_images_per_prompt = self.attributes.generation_params.get("num_images_per_prompt", 1)
        self.generator = self._make_generator()

        if self.attributes.enable_model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()

    def _make_generator(self) -> torch.Generator | list[torch.Generator]:
        """Creates and seeds a PyTorch generator or a list of generators based on the provided
        seed value(s).

        Returns:
            torch.Generator | list[torch.Generator]: A generator or a list of generators for
            random number generation.
        """
        if isinstance(self.attributes.seed, int):
            log_msg = f"Using seed: {self.attributes.seed}"
            self.logger.info(log_msg)
            return torch.Generator(device=self.attributes.device).manual_seed(self.attributes.seed)
        if isinstance(self.attributes.seed, list):
            random_seeds = torch.randint(
                0,
                2**32 - 1,
                (max(0, self.num_images_per_prompt - len(self.attributes.seed)),),
            ).tolist()
            seeds = self.attributes.seed[: self.num_images_per_prompt] + random_seeds
        else:
            seeds = torch.randint(0, 2**32 - 1, (self.num_images_per_prompt,)).tolist()

        log_msg = f"Using seeds: {seeds}"
        self.logger.info(log_msg)

        return [torch.Generator(device=self.attributes.device).manual_seed(s) for s in seeds]

    def _make_pipeline(self) -> DiffusionPipeline:
        """Creates and configures a diffusion pipeline for the generative task.

        This method initializes the diffusion pipeline using the specified model path, torch data type,
        and cache directory. It also ensures the pipeline is moved to the appropriate device (CPU or GPU).

        Returns:
            DiffusionPipeline: A configured pipeline object ready for image generation.
        """
        pipeline_class = self._pipeline_class()
        return pipeline_class.from_pretrained(
            self.attributes.model_path,
            torch_dtype=self.TORCH_DTYPE.get(self.attributes.torch_dtype),
            cache_dir=self.attributes.model_cache_dir,
        ).to(self.attributes.device)

    @staticmethod
    @abstractmethod
    def _pipeline_class() -> DiffusionPipeline:
        """Returns the specific diffusion pipeline class to be used for the task.

        This method must be implemented by subclasses to specify the type of pipeline (e.g., text-to-image,
        inpainting, etc.) to be used.

        Returns:
            DiffusionPipeline: The class type of the specific pipeline to be used.
        """

    def _generate_images(
        self,
        inputs: dict[str, np.ndarray | list[np.ndarray]] | None = None,
        output_attribute: Literal["images", "frames"] = "images",
    ) -> list[np.ndarray]:
        """Generates images or frames using the configured diffusion pipeline.

        This method runs the pipeline with the provided inputs and generation parameters. It supports
        generating either images or frames, depending on the specified `output_attribute`.

        Args:
            inputs (dict[str, np.ndarray  |  list[np.ndarray]], optional): A dictionary of input data for the pipeline.
                Keys should match the expected input names for the pipeline (e.g., "image", "mask"). Defaults to {}.
            output_attribute (Literal["images", "frames"], optional): Specifies whether to extract
                "images" or "frames", depending on the pipeline class used. Defaults to "images".

        Returns:
            list[np.ndarray]: A list of generated images or frames as numpy arrays in uint8 format (0-255).
        """
        if inputs is None:
            inputs = {}
        output = self.pipeline(
            **inputs,
            **self.attributes.generation_params,
            generator=self.generator,
            output_type="np",
        )
        generated_images = output.images if output_attribute == "images" else output.frames[0]

        return [(image * 255).clip(0, 255).astype(np.uint8) for image in generated_images]

    def _update_images_in_container(self, container: DataContainer, new_packets: list[ImagePacket]) -> None:
        """Updates the container with new image packets based on the `overwrite_images` attribute.

        This method either replaces the existing images in the container with the new packets or
        appends the new packets to the existing ones, depending on the value of `overwrite_images`.

        Args:
            container (DataContainer): The data container to update. Existing image content will
            be replaced with the new packets if `overwrite_images` is True.
            new_packets (list[ImagePacket]): List of new image packets to add to the container.
        """
        if self.attributes.overwrite_images:
            container.images = new_packets
        else:
            container.images.extend(new_packets)

    def _clear_memory(self) -> None:
        """Clears memory to free up resources.

        This method performs garbage collection and clears GPU memory (if applicable) to prevent memory leaks
        and ensure efficient resource usage.
        """
        gc.collect()
        if self.attributes.device == "cuda":
            torch.cuda.empty_cache()
