# -*- coding: utf-8 -*-

from diffusers import AutoPipelineForText2Image
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket

from sinapsis_huggingface_diffusers.templates.base_diffusers import BaseDiffusers


class TextToImageDiffusers(BaseDiffusers):
    """This class implements a specific template for text-to-image generation using Hugging Face's
    diffusers. The `TextToImageDiffusers` class inherits from the base `BaseDiffusers` class
    to define how to handle text-based prompts for generating images.

    Parameters to be used inside generation_params can be seen at HuggingFace documentation:
    https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: TextToImageDiffusers
      class_name: TextToImageDiffusers
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/diffusers/model'
        model_cache_dir: /path/to/cache/dir
        device: 'cuda'
        torch_dtype: float16
        enable_model_cpu_offload: false
        generation_params: {}
    """

    @staticmethod
    def _pipeline_class() -> AutoPipelineForText2Image:
        """Returns the `AutoPipelineForText2Image` class to be used for text-to-image generation.

        This method specifies the pipeline class required for text-to-image tasks. It ensures the
        correct pipeline is used for generating images from text prompts.

        Returns:
            AutoPipelineForText2Image: The class reference for `AutoPipelineForText2Image`.
        """
        return AutoPipelineForText2Image

    @staticmethod
    def _set_packet_sources(image_packets: list[ImagePacket]) -> None:
        """Sets the `source` field for each image packet.

        This method processes the image packets by setting their `source` field to their unique ID.

        Args:
            image_packets (list[ImagePacket]): List of image packets to process.
        """
        for image_packet in image_packets:
            image_packet.source = str(image_packet.id)

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the text-to-image generation process and updates the container with the results.

        This method generates images using the configured diffusion pipeline, updates the provided
        `DataContainer` with the generated images, and clears memory to free up resources.

        Args:
            container (DataContainer): The data container to update with the generated images.

        Returns:
            DataContainer: The updated data container containing the generated images.
        """
        generated_images = self._generate_images()
        image_packets = [ImagePacket(content=image) for image in generated_images]
        self._set_packet_sources(image_packets)
        self._update_images_in_container(container, image_packets)
        self._clear_memory()

        return container
