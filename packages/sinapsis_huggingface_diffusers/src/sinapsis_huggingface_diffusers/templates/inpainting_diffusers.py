# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Literal

import cv2
import numpy as np
from diffusers import AutoPipelineForInpainting
from sinapsis_core.data_containers.data_packet import ImageAnnotations, ImagePacket
from sinapsis_core.template_base.template import TemplateAttributeType

from sinapsis_huggingface_diffusers.templates.base_diffusers import (
    BaseDiffusersAttributes,
)
from sinapsis_huggingface_diffusers.templates.image_to_image_diffusers import (
    ImageToImageDiffusers,
)


class InpaintingDiffusersAttributes(BaseDiffusersAttributes):
    """Attributes specific to the InpaintingDiffusers template.

    Attributes:
        inpainting_mode (Literal["bbox", "mask"]): Determines how the inpainting mask is generated.
            - "bbox": Generate masks based on bounding box annotations.
            - "mask": Use segmentation masks or polygons to generate the mask.
        dilation_radius (int | None): Radius of the surrounding area (in pixels) to include
            around the mask. If None, no dilation is applied. Defaults to None.
        preserve_outside_content (bool): If True, preserves the content outside the mask by
            blending the new content with the old content. Defaults to False.
    """

    inpainting_mode: Literal["bbox", "mask"]
    dilation_radius: int | None = None
    preserve_outside_content: bool = False


class InpaintingDiffusers(ImageToImageDiffusers):
    """This class implements a specific template for inpainting tasks using Hugging Face's diffusers.
    The `InpaintingDiffusers` class inherits from `ImageToImageDiffusers` and extends its capabilities
    to handle inpainting using bounding boxes to create masks or directly using segmentation masks.

    Parameters to be used inside generation_params can be seen at HuggingFace documentation:
    https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: InpaintingDiffusers
      class_name: InpaintingDiffusers
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/diffusers/model'
        model_cache_dir: /path/to/cache/dir
        device: 'cuda'
        torch_dtype: float16
        enable_model_cpu_offload: false
        generation_params: {}
        overwrite_images: false
        inpainting_mode: 'bbox'
        dilation_radius: 2
        preserve_outside_content: false

    """

    AttributesBaseModel = InpaintingDiffusersAttributes

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        if self.attributes.preserve_outside_content and self.attributes.dilation_radius is None:
            raise ValueError("Need to specify a dilation_radius if preserve_outside_content=True")

    @staticmethod
    def _pipeline_class() -> AutoPipelineForInpainting:
        """Returns the `AutoPipelineForInpainting` class to be used for inpainting generation.

        Returns:
            AutoPipelineForInpainting: The class reference for `AutoPipelineForInpainting`.
        """
        return AutoPipelineForInpainting

    @staticmethod
    def _initialize_mask(image_packet: ImagePacket) -> np.ndarray:
        """
        Initializes a binary mask for inpainting.

        Args:
            image_packet (ImagePacket): The input image packet containing image data.

        Returns:
            np.ndarray: A binary mask initialized to zeros (empty) or ones (full).
        """
        h, w = image_packet.content.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    @staticmethod
    def _generate_mask_from_bboxes(image_packet: ImagePacket, mask: np.ndarray) -> np.ndarray:
        """Generates a binary mask based on bounding boxes in the image annotations.

        Args:
            image_packet (ImagePacket): The input image packet containing bounding box annotations.
            mask (np.ndarray): The binary mask to update.

        Returns:
            np.ndarray: The updated binary mask with bounding box regions marked for inpainting.
        """
        for annotation in image_packet.annotations:
            if annotation.bbox:
                x1, y1 = int(annotation.bbox.x), int(annotation.bbox.y)
                x2, y2 = (
                    int(annotation.bbox.x + annotation.bbox.w),
                    int(annotation.bbox.y + annotation.bbox.h),
                )
                mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def _generate_mask_from_segmentation(image_packet: ImagePacket, mask: np.ndarray) -> np.ndarray:
        """Generates a binary mask based on segmentation data in the image annotations.

        Args:
            image_packet (ImagePacket): The input image packet containing segmentation annotations.
            mask (np.ndarray): The binary mask to update.

        Returns:
            np.ndarray: The updated binary mask with segmentation regions marked for inpainting.
        """
        for annotation in image_packet.annotations:
            segmentation = annotation.segmentation
            if segmentation:
                if segmentation.mask is not None:
                    mask = segmentation.mask
                    break
                if segmentation.polygon is not None:
                    for poly in segmentation.polygon:
                        polygon = np.array(poly, dtype=np.int32)
                        cv2.fillPoly(mask, [polygon], 255)

        return mask

    def _generate_mask(self, image_packet: ImagePacket) -> np.ndarray:
        """Generates a binary mask for inpainting based on the configured mode.

        If no annotations or the required annotation type is missing, a mask covering the
        entire image is returned, and a warning is logged.

        Args:
            image_packet (ImagePacket): The input image packet containing data and annotations.

        Returns:
            np.ndarray: The binary mask marking regions for inpainting.
        """
        mask = self._initialize_mask(image_packet)
        annotations = image_packet.annotations or []
        mode = self.attributes.inpainting_mode

        has_valid_annotations = any(ann.bbox if mode == "bbox" else ann.segmentation for ann in annotations)

        if not has_valid_annotations:
            mask.fill(255)
            log_msg = f"No valid '{mode}' annotations. The entire image will be replaced."
            self.logger.warning(log_msg)
            return mask

        if mode == "bbox":
            return self._generate_mask_from_bboxes(image_packet, mask)

        if mode == "mask":
            return self._generate_mask_from_segmentation(image_packet, mask)

        return mask

    def preprocess_inputs(self, image_packet: ImagePacket) -> dict[str, np.ndarray | list[np.ndarray]]:
        """Prepares the input image and mask for the inpainting pipeline.

        This method extends the `ImageToImageDiffusers` preprocessing by adding a binary
        mask to the dictionary.

        Args:
            image_packet (ImagePacket): The input image packet with content and annotations.

        Returns:
            dict[str, np.ndarray | list[np.ndarray]]: A dictionary with the preprocessed image(s)
            and mask(s).
        """
        inputs = super().preprocess_inputs(image_packet)
        mask = self._generate_mask(image_packet)
        inputs.update({"mask_image": ([mask] * self.num_images_per_prompt if self.num_images_per_prompt > 1 else mask)})
        return inputs

    @staticmethod
    def _update_annotations(
        annotations: ImageAnnotations,
        old_w: int,
        old_h: int,
        new_w: int,
        new_h: int,
    ) -> None:
        """Updates the bounding boxes and segmentation annotations to fit the new image dimensions.

        Args:
            annotations (ImageAnnotations): The annotations to update, containing bounding boxes and
                segmentation data.
            old_w (int): The original width of the image.
            old_h (int): The original height of the image.
            new_w (int): The new width of the image.
            new_h (int): The new height of the image.
        """
        for ann in annotations:
            if ann.bbox:
                ann.bbox.x = int(ann.bbox.x * new_w / old_w)
                ann.bbox.y = int(ann.bbox.y * new_h / old_h)
                ann.bbox.w = int(ann.bbox.w * new_w / old_w)
                ann.bbox.h = int(ann.bbox.h * new_h / old_h)

            if ann.segmentation:
                if ann.segmentation.mask is not None:
                    ann.segmentation.mask = cv2.resize(
                        ann.segmentation.mask,
                        (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                if ann.segmentation.polygon is not None:
                    ann.segmentation.polygon = [
                        [
                            (
                                int(pt[0] * new_w / old_w),
                                int(pt[1] * new_h / old_h),
                            )
                            for pt in polygon
                        ]
                        for polygon in ann.segmentation.polygon
                    ]

    def _update_old_packets(self, old_packets: list[ImagePacket], new_packets: list[ImagePacket]) -> list[ImagePacket]:
        """Resizes the old packets to match the dimensions of the new packets and updates their
        annotations.

        Args:
            old_packets (list[ImagePacket]): The original image packets to be updated.
            new_packets (list[ImagePacket]): The newly generated image packets with the target
                dimensions.

        Returns:
            list[ImagePacket]: A list of updated image packets, resized and with updated
                annotations.
        """
        updated_old_packets = []

        for old_packet, new_packet in zip(old_packets, new_packets):
            new_h, new_w = new_packet.content.shape[:2]
            old_h, old_w = old_packet.content.shape[:2]

            updated_packet = deepcopy(old_packet)
            updated_packet.content = cv2.resize(updated_packet.content, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self._update_annotations(updated_packet.annotations, old_w, old_h, new_w, new_h)

            updated_old_packets.append(updated_packet)

        return updated_old_packets

    def _blend_images(self, old_packet: ImagePacket, new_packet: ImagePacket, mask: np.ndarray) -> np.ndarray:
        """Blends content from the old image packet with the new image packet based on the mask.

        Args:
            old_packet (ImagePacket): The original image packet.
            new_packet (ImagePacket): The newly generated image packet.
            mask (np.ndarray): A binary mask indicating regions to update (1: update, 0: preserve).

        Returns:
            np.ndarray: The blended content for the new packet.
        """
        min_dimension = min(mask.shape)
        kernel_size = min(self.attributes.dilation_radius * 2 + 1, min_dimension)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        dilated_mask = np.clip(dilated_mask, 0, 1).astype(bool)
        blended_content = old_packet.content.copy()
        blended_content[dilated_mask] = new_packet.content[dilated_mask]

        return blended_content

    def post_processing_packets(
        self,
        new_packets: list[ImagePacket],
        old_packets: list[ImagePacket],
    ) -> tuple[list[ImagePacket], list[ImagePacket]]:
        """Processes newly generated image packets for the inpainting pipeline.

        Extends the functionality of the `ImageToImageDiffusers` class by blending the content
        of new image packets with corresponding old packets based on the `blend_threshold`
        attribute. Updates annotations from the old packets to the new packets, including setting
        the `label_str` field with the prompt used during generation.

        Args:
            new_packets (list[ImagePacket]): The newly generated image packets.
            old_packets (list[ImagePacket]): The existing image packets in the container.

        Returns:
            tuple[list[ImagePacket], list[ImagePacket]]: The processed list of new image packets
                with blended content and updated annotations, and the unchanged list of old
                image packets.
        """
        new_packets, old_packets = super().post_processing_packets(new_packets, old_packets)
        if not old_packets:
            return new_packets, old_packets

        old_packets = self._update_old_packets(old_packets, new_packets)

        for new_packet, old_packet in zip(new_packets, old_packets):
            if self.attributes.preserve_outside_content:
                if np.all(new_packet.content == 0):
                    continue
                mask = self._generate_mask(old_packet)
                new_packet.content = self._blend_images(old_packet, new_packet, mask)

            if old_packet.annotations:
                new_packet.annotations = old_packet.annotations
                for ann in new_packet.annotations:
                    ann.label_str = str(self.attributes.generation_params.get("prompt"))

        return new_packets, old_packets
