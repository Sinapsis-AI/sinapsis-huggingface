# -*- coding: utf-8 -*-
import numpy as np
import torch
from sinapsis_core.data_containers.annotations import BoundingBox, ImageAnnotations
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket
from sinapsis_core.template_base import TemplateAttributeType
from sinapsis_data_visualization.helpers.detection_utils import bbox_xyxy_to_xywh
from sinapsis_huggingface_transformers.templates.pali_gemma.pali_gemma_base import (
    PaliGemmaBase,
    PaliGemmaBaseAttributes,
)
from transformers.generation.utils import GenerateOutput


class PaliGemmaInferenceAttributes(PaliGemmaBaseAttributes):
    """Configuration attributes for PaliGemma inference.

    Attributes:
        prompt (str): Prompt to run the inference (default: "<image>caption en")

    The <image> token is essential as it serves as a marker that tells the model where to look at the image
    when processing the input. This token enables the model to understand the relationship between the visual
    and textual components during processing.

    Example prompts:
        - "<image>caption en" -> Generates a basic caption in English
        - "<image>What objects can you see in this image?" -> Lists objects in the image
    """

    prompt: str = "<image>caption en"


class PaliGemmaInference(PaliGemmaBase):
    """Implementation of PaliGemma inference pipeline for image processing and caption generation.

    This class handles the inference process for PaliGemma models, including image processing,
    caption generation, and annotation creation. It supports both basic captioning and
    detection/segmentation tasks.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: PaliGemmaInference
      class_name: PaliGemmaInference
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/paligemma/model'
        processor_path: '`/path/to/processor'
        model_cache_dir: /path/to/cache/dir
        device: 'cuda'
        max_new_tokens: 200
        torch_dtype: float16
        prompt: <image> caption en

    """

    AttributesBaseModel = PaliGemmaInferenceAttributes
    INPUT_IDS = "input_ids"

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.prompt = self.attributes.prompt

    def _prepare_inputs(self, image_content: np.ndarray) -> dict:
        """Prepares the input for model inference by processing the image and text prompt.

        Args:
            image_content (np.ndarray): Raw image content to be processed as a numpy array

        Returns:
            dict: Processed inputs containing:
                - input_ids (torch.Tensor): Token IDs for the text prompt and image tokens
                - attention_mask (torch.Tensor): Binary mask indicating valid input positions (1s)
                - pixel_values (torch.Tensor): Processed image tensor with normalized pixel values
                    in shape (batch_size, channels, height, width)

        Note:
            - The format of the returns it's because uses PyTorch tensors (return_tensors="pt")
        """

        return self.processor(
            images=image_content,
            text=self.prompt,
            return_tensors="pt",
        ).to(self.attributes.device)

    def _generate_caption(self, inputs: dict) -> torch.Tensor:
        """Generates caption using the model.

        Args:
            inputs (dict): Processed model inputs for the processor, including input IDs of the image and prompt

        Returns:
            GeneratedCaptionOutput: A structured output containing:
                - sequences: tensor with token IDs of the generated sequence
                - scores: tuple of tensors with token prediction scores for each generation step
                - logits: optional tensor with raw logits (None in this configuration)
                - attentions: optional attention weights (None in this configuration)
                - hidden_states: optional hidden states (None in this configuration)
                - past_key_values: tuple of tensors containing past keys/values for attention mechanism

        Configuration parameters:
            - max_new_tokens: Maximum number of new tokens to generate
            - return_dict_in_generate: Returns output as a structured dictionary
            - output_scores: Includes prediction scores in the output
        """
        with torch.no_grad():
            return self.model.generate(
                **inputs,
                max_new_tokens=self.attributes.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

    @staticmethod
    def _calculate_confidence_score(outputs: GenerateOutput) -> float:
        """Calculates the confidence score from model generation outputs.

        The confidence score is computed as the mean of the highest probability
        for each generated token in the sequence.

        Args:
            outputs (GenerateOutput): Model generation output containing scores
                for each generated token

        Returns:
            float: Average confidence score across all generated tokens
        """
        scores = torch.stack(outputs.scores)
        probs = torch.softmax(scores, dim=-1)
        token_confidences = torch.max(probs, dim=-1).values
        return float(torch.mean(token_confidences).cpu())

    def _decode_caption(self, outputs: GenerateOutput, input_len: int) -> str:
        """Decodes the model output sequences into readable caption text.

        Args:
            outputs (GenerateOutput): Model generation output containing the
                generated token sequences
            input_len (int): Length of the input sequence to skip initial tokens

        Returns:
            str: Decoded caption text with special tokens removed
        """
        return self.processor.decode(outputs.sequences[0][input_len:], skip_special_tokens=True)

    def _create_annotation(
        self, caption: str, confidence: float, image_shape: tuple[int, ...]
    ) -> list[ImageAnnotations]:
        """Creates image annotations from the generated caption.

        Args:
            caption (str): Generated caption text
            confidence (float): Confidence score for the prediction
            image_shape (tuple[int, ...]): Shape of the input image

        Returns:
            list[ImageAnnotations]: List containing annotation with caption information
        """

        _, _ = self, image_shape
        return [ImageAnnotations(text=caption, confidence_score=confidence)]

    def _process_single_image(self, image_packet: ImagePacket) -> None:
        """Processes a single image through the inference pipeline.

        Args:
            image_packet (ImagePacket): Container with image data and metadata

        Returns:
            None: Modifies the image_packet in place by adding annotations
        """
        inputs = self._prepare_inputs(image_packet.content)
        outputs = self._generate_caption(inputs)
        input_len = inputs[self.INPUT_IDS].shape[-1]
        caption = self._decode_caption(outputs, input_len)
        confidence = self._calculate_confidence_score(outputs)
        annotations = self._create_annotation(caption, confidence, image_packet.content.shape)
        image_packet.annotations.extend(annotations)

    def _format_text_for_prompt(self, text: str) -> str:
        """Formats the incoming text appropriately for the current task type.
            Base implementation returns the text as-is, subclasses may override
        to apply task-specific formatting.
            Args:
            text (str): Raw text content
            Returns:
            str: Formatted prompt text
        """
        _ = self
        return text

    def process_from_text_packet(self, container: DataContainer) -> None:
        """
        Extract prompts from the received list of text packets and use them to perform inference in each received image
        packet.

        Args:
            container (DataContainer): Data-container with text and image packets to be processed.
        """
        for text_packet in container.texts:
            self.prompt = self._format_text_for_prompt(text_packet.content)
            if container.images:
                for image_packet in container.images:
                    self._process_single_image(image_packet)

    def process_from_prompt(self, container: DataContainer) -> None:
        """
        Perform inference in each received image packet using the prompt defined in template attributes.

        Args:
            container (DataContainer): Data-container with image packets to be processed.
        """
        if container.images:
            for image_packet in container.images:
                self._process_single_image(image_packet)

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the inference pipeline on a batch of images.

        If text packets are present, uses each text as input for prompt formatting.
        If no text packets exist, uses the default prompt from attributes.

        Args:
            container (DataContainer): Container with text and image packets

        Returns:
            DataContainer: Processed container with added annotations
        """
        if container.texts:
            self.process_from_text_packet(container)
        else:
            self.process_from_prompt(container)

        return container

    @staticmethod
    def create_bbox_annotation(coords: tuple[float, ...], label: str, confidence: float) -> ImageAnnotations:
        """Creates bounding box annotation from coordinates and metadata.

        Args:
            coords (tuple[float, ...]): Coordinates (x0, y0, x1, y1)
            label (str): Label for the detected object
            confidence (float): Confidence score for the detection

        Returns:
            ImageAnnotations: Annotation object with bounding box information
        """
        x0, y0, x1, y1 = coords
        x, y, w, h = bbox_xyxy_to_xywh([x0, y0, x1, y1])
        return ImageAnnotations(
            label_str=label,
            confidence_score=confidence,
            bbox=BoundingBox(x=x, y=y, w=w, h=h),
        )
