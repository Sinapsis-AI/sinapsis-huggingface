# -*- coding: utf-8 -*-

import os
from copy import deepcopy
from typing import Literal

import numpy as np
from diffusers import AutoPipelineForImage2Image
from PIL import Image
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket
from sinapsis_core.template_base.template import TemplateAttributeType

from sinapsis_huggingface_diffusers.templates.base_diffusers import BaseDiffusers


class ImageToImageDiffusers(BaseDiffusers):
    """Template for the image-to-image generative task using diffusers.

    This class extends the base `BaseDiffusers` class to provide an implementation
    for the specific  image-to-image generation using HuggingFace's Diffusers.

    Parameters to be used inside generation_params can be seen at HuggingFace documentation:
    https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: ImageToImageDiffusers
      class_name: ImageToImageDiffusers
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/diffusers/model'
        model_cache_dir: /path/to/sinapsis/cache/dir
        device: 'cuda'
        torch_dtype: float16
        enable_model_cpu_offload: false
        generation_params: {}
        overwrite_images: false



    """

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.requires_pil = False
        self.output_attribute: Literal["images", "frames"] = "images"
        self.num_duplicates = self.num_images_per_prompt

    @staticmethod
    def _pipeline_class() -> AutoPipelineForImage2Image:
        """Returns the `AutoPipelineForImage2Image` class to be used for image-to-image generation.

        Returns:
            AutoPipelineForImage2Image: The class reference for `AutoPipelineForImage2Image`.
        """
        return AutoPipelineForImage2Image

    def _convert_image_format(self, image_packet: ImagePacket) -> np.ndarray | Image.Image:
        """Converts the input image into the appropriate format for the pipeline.

        The format depends on the `requires_pil` attribute:
        - If `requires_pil` is True, the image is converted to a PIL Image.
        - If `requires_pil` is False, the image is normalized as a NumPy array to the range [0, 1].

        Args:
            image_packet (ImagePacket): The input image packet.

        Returns:
            np.ndarray | Image.Image: The converted image, either as a normalized NumPy array or a
                PIL Image.
        """
        if self.requires_pil:
            return Image.fromarray(image_packet.content)
        return image_packet.content / 255.0

    def preprocess_inputs(self, image_packet: ImagePacket) -> dict[str, np.ndarray | list[np.ndarray]]:
        """Prepares the input image for the image-to-image pipeline.

        Normalizes the image to the range [0, 1]. Duplicates the image if multiple images per
        prompt are required.

        Args:
            image_packet (ImagePacket): The input image packet.

        Returns:
            dict[str, np.ndarray | list[np.ndarray]]: A dictionary with the preprocessed image(s).
        """
        input_image = self._convert_image_format(image_packet)
        inputs = {
            "image": ([input_image] * self.num_images_per_prompt if self.num_images_per_prompt > 1 else input_image)
        }
        return inputs

    def post_processing_packets(
        self,
        new_packets: list[ImagePacket],
        old_packets: list[ImagePacket],
    ) -> tuple[list[ImagePacket], list[ImagePacket]]:
        """Processes newly generated image packets for the image-to-image pipeline.

        Associates each new image packet with its corresponding old packet by updating the `source`
        field. The `source` of each new packet is set to the `source` of its corresponding old
        packet, appended with the new packet's unique ID. If multiple images are generated per
        prompt, the old packets are duplicated to match the number of new packets.

        Args:
            new_packets (list[ImagePacket]): The newly generated image packets.
            old_packets (list[ImagePacket]): The existing image packets in the container.

        Returns:
            tuple[list[ImagePacket], list[ImagePacket]]: The processed list of new image packets
                with updated `source` fields and the duplicated list of old image packets.
        """
        if old_packets:
            old_packets = [deepcopy(old_packet) for old_packet in old_packets for _ in range(self.num_duplicates)]
            for new_packet, old_packet in zip(new_packets, old_packets):
                filename = os.path.basename(old_packet.source).split(".")[0]
                new_packet.source = f"{filename}_{new_packet.id!s}"
        return new_packets, old_packets

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the image generation process and updates the container with the results.

        This method processes each image in the container, generates new images using the configured
        pipeline, and updates the container with the generated images. It also clears memory to
        free up resources.

        Args:
            container (DataContainer): The data container containing the input images.

        Returns:
            DataContainer: The updated data container containing the generated images.
        """
        if not container.images:
            return container

        old_packets = deepcopy(container.images)
        all_generated_images: list[np.ndarray] = []

        for image_packet in container.images:
            inputs = self.preprocess_inputs(image_packet)
            generated_images = self._generate_images(inputs, self.output_attribute)
            all_generated_images.extend(generated_images)

        new_packets = [ImagePacket(content=image) for image in all_generated_images]
        processed_packets, _ = self.post_processing_packets(new_packets, old_packets)
        self._update_images_in_container(container, processed_packets)
        self._clear_memory()

        return container
