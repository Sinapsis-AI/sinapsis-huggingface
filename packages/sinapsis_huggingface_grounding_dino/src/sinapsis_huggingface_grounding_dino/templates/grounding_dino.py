# -*- coding: utf-8 -*-


from typing import Any, Literal

import torch
from sinapsis_core.data_containers.annotations import BoundingBox, ImageAnnotations
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket
from sinapsis_core.template_base import Template, TemplateAttributes, TemplateAttributeType
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    GroundingDinoForObjectDetection,
    PreTrainedModel,
)

from sinapsis_huggingface_grounding_dino.helpers.grounding_dino_keys import GroundingDINOKeys


class GroundingBaseAttributes(TemplateAttributes):
    """GroundingDINOAttributes defines the base configuration attributes for the GroundingDINO
    classes.

    Attributes:
        model_path (str): Specifies the model identifier or file path for the base GroundingDINO
            model. Example:
            "IDEA-Research/grounding-dino-tiny".
        model_cache_dir (str): Directory where model files are or will be stored. Defaults to
            "SINAPSIS_CACHE_DIR".
        inference_mode (Literal["object_detection", "zero_shot"]): Specifies the mode for model
            inference. "object_detection" enables direct object detection, while "zero_shot"
            enables zero-shot inference for unseen classes.
        threshold (float): Threshold for box detection. Defaults to 0.25.
        text_threshold (float): Threshold for text detection. Defaults to 0.25.
        device (Literal["cuda", "cpu"]): Device to be used for inference.
    """

    model_path: str
    model_cache_dir: str = str(SINAPSIS_CACHE_DIR)
    inference_mode: Literal["object_detection", "zero_shot"]
    threshold: float = 0.25
    text_threshold: float = 0.25
    device: Literal["cuda", "cpu"]


class GroundingDINO(Template):
    """Base class for Grounding DINO models.

    This module contains the foundational class for implementing object detection using
    the Grounding DINO model, leveraging transformers and PyTorch. It provides essential
    methods for running inference, formatting results, and creating annotations, which
    can be extended by other specialized classes related to Grounding DINO.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: GroundingDINO
      class_name: GroundingDINO
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/model'
        model_cache_dir: /path/to/cache/dir
        inference_mode: 'object_detection'
        threshold: 0.25
        text_threshold: 0.25
        device: 'cuda'
        text_input: 'object to detect'

    """

    CATEGORY = "Grounding DINO"
    KEYS = GroundingDINOKeys()

    class AttributesBaseModel(GroundingBaseAttributes):
        """GroundingDINOAttributes defines the configuration attributes for the GroundingDINO class.

        Attributes:
            text_input (str): Input text for GroundingDINO processing.
        """

        text_input: str

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """Initializes the GroundingDINO class with the provided attributes.

        Args:
            attributes (dict[str, Any]): Dictionary containing configuration parameters.
        """
        super().__init__(attributes)
        self.device = self.attributes.device
        self.processor = AutoProcessor.from_pretrained(
            self.attributes.model_path, cache_dir=self.attributes.model_cache_dir
        )
        self.model = self._set_model().to(self.device)
        self.max_tokens = self.processor.tokenizer.model_max_length
        self.text_input = self.validate_and_format_text_input(self.attributes.text_input)

    def _set_model(self) -> PreTrainedModel:
        """Loads the specific model variant based on the configured inference mode.

        Returns:
            PreTrainedModel: The pretrained model instance, loaded according to the inference mode.
        """
        if self.attributes.inference_mode == "object_detection":
            return GroundingDinoForObjectDetection.from_pretrained(
                self.attributes.model_path, cache_dir=self.attributes.model_cache_dir
            )
        return AutoModelForZeroShotObjectDetection.from_pretrained(
            self.attributes.model_path, cache_dir=self.attributes.model_cache_dir
        )

    def validate_and_format_text_input(self, text_input: str) -> str:
        """Validates and formats the text input for consistency.

        Args:
            text_input (str): The input text specifying object classes.

        Returns:
            str: A validated and formatted version of the input text.
        """
        delimiter = self.KEYS.CLASS_DELIMITER

        if delimiter not in text_input:
            raise ValueError(
                f"Invalid text_input format '{text_input}': Expected at least one '{delimiter}' "
                "to separate object names."
            )

        formatted_input = " ".join(text_input.split())
        formatted_input = f"{delimiter} ".join(part.strip() for part in formatted_input.split(delimiter))

        if not formatted_input.endswith(f"{delimiter} "):
            formatted_input += delimiter

        return formatted_input.strip()

    def _run_inference(self, image_packet: ImagePacket) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Runs inference on a given image packet.

        Args:
            image_packet (ImagePacket): Image data to be processed.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: Outputs and inputs of the processor, respectively.
        """
        inputs = self.processor(
            images=image_packet.content,
            text=self.text_input,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs, inputs

    def _post_process(
        self,
        outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        image_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Processes model outputs to extract and format detection results.

        This method uses the processor's post-processing function to generate
        detection results such as bounding boxes, confidence scores, and class labels
        for the objects detected in the input image.

        Args:
            outputs (dict[str, torch.Tensor]): The raw output tensors from the model, containing
                information required for generating detections (e.g., bounding box coordinates,
                scores).
            inputs (dict[str, torch.Tensor]): Input tensors provided to the model, including input
                IDs for identifying objects.
            image_size (tuple[int, int]): The dimensions (height, width) of the input image, used
                to scale the bounding boxes to the image size.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each containing detection information
                for an identified object, including bounding boxes, labels, and confidence scores.
        """
        detections: list[dict[str, Any]] = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs[self.KEYS.INPUT_IDS],
            threshold=self.attributes.threshold,
            text_threshold=self.attributes.text_threshold,
            target_sizes=[image_size],
        )
        return detections

    def _format_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Formats the results by converting scores and boxes to CPU tensors and converting labels
        to strings.

        Args:
            results (list[dict[str, Any]]): List of detection results containing scores, labels,
                and boxes.

        Returns:
            list[dict[str, Any]]: A list of formatted results with tensors moved to the CPU, labels converted to
                strings.
        """
        formatted_results = []
        for result in results:
            formatted_result = {
                self.KEYS.CONFIDENCE_SCORE: result[self.KEYS.CONFIDENCE_SCORE].cpu(),
                self.KEYS.LABELS: [
                    str(label) if isinstance(label, str) else str(label.item()) for label in result[self.KEYS.LABELS]
                ],
                self.KEYS.BBOXES: result[self.KEYS.BBOXES].cpu(),
            }

            formatted_results.append(formatted_result)

        return formatted_results

    def get_class_names(self) -> list[str]:
        """
        Produce a list of class names from the given text input.

        Returns:
            list[str]: List of class names.
        """
        return [cls.strip() for cls in self.text_input.split(self.KEYS.CLASS_DELIMITER) if cls.strip()]

    def _filter_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filters results to exclude annotations with empty or whitespace-only labels,
        and discards labels not found in the provided class names.

        Args:
            results (list[dict[str, Any]]): List of detection results containing scores, labels,
                and boxes.

        Returns:
            list[dict[str, Any]]: A list of filtered results with empty or invalid labels removed.
        """
        class_names = self.get_class_names()
        filtered_results = []

        for result in results:
            valid_labels = [idx for idx, label in enumerate(result[self.KEYS.LABELS]) if label.strip() in class_names]

            if valid_labels:
                filtered_result = {
                    self.KEYS.CONFIDENCE_SCORE: result[self.KEYS.CONFIDENCE_SCORE][valid_labels],
                    self.KEYS.LABELS: [result[self.KEYS.LABELS][i] for i in valid_labels],
                    self.KEYS.BBOXES: result[self.KEYS.BBOXES][valid_labels],
                }

                filtered_results.append(filtered_result)

        return filtered_results

    def _prepare_results(
        self,
        outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        image_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Prepare and filter results from the model outputs.

        Args:
            outputs (dict[str, torch.Tensor]): Model outputs from inference.
            inputs (dict[str, torch.Tensor]): Input data fed to the model processor.
            image_size (tuple[int, int]): Size of the processed image.

        Returns:
            list[dict[str, Any]]: Final post-processed, filtered and formatted results from the model.
        """
        results = self._post_process(outputs, inputs, image_size)
        formatted_results = self._format_results(results)
        return self._filter_results(formatted_results)

    def _create_annotations(self, image_packet: ImagePacket, results: list[dict[str, Any]]) -> None:
        """Creates annotation objects for the image packet based on filtered detection results.

        Args:
            image_packet (ImagePacket): Container holding the processed image and metadata.
            list[dict[str, Any]]: Detection results including bounding boxes, labels, and scores.
        """
        new_annotations: list[ImageAnnotations] = []
        for result in results:
            for idx, bbox in enumerate(result[self.KEYS.BBOXES].numpy()):
                xmin, ymin, xmax, ymax = bbox
                bounding_box = BoundingBox(
                    x=xmin,
                    y=ymin,
                    w=xmax - xmin,
                    h=ymax - ymin,
                )

                new_annotations.append(
                    ImageAnnotations(
                        label_str=result[self.KEYS.LABELS][idx],
                        confidence_score=float(result[self.KEYS.CONFIDENCE_SCORE][idx]),
                        bbox=bounding_box,
                    )
                )
        image_packet.annotations = new_annotations

    def _process_image_packet(self, image_packet: ImagePacket) -> None:
        """Processes a single image packet by preparing detection results and creating annotations.

        Args:
            image_packet (ImagePacket): An image packet to be processed, producing detection
                results and corresponding annotations.
        """
        outputs, inputs = self._run_inference(image_packet)
        image_size = image_packet.content.shape[:2]
        detections = self._prepare_results(outputs, inputs, image_size)
        self._create_annotations(image_packet, detections)

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the detection pipeline for a container of images by processing each image
        packet.

        Args:
            container (DataContainer): The container holding multiple image packets for processing.

        Returns:
            DataContainer: The container with updated annotations for each image packet.
        """

        for image_packet in container.images:
            self._process_image_packet(image_packet)

        return container
