# -*- coding: utf-8 -*-
"""
GroundingDINO Zero-Shot Classification Module.

This module provides an implementation of a zero-shot classification system
using the GroundingDINO model. The classification pipeline allows for image
classification tasks based on a predefined set of classes or text input.
"""

from typing import Any

from sinapsis_core.data_containers.annotations import ImageAnnotations
from sinapsis_core.data_containers.data_packet import ImagePacket

from sinapsis_huggingface_grounding_dino.templates.grounding_dino import GroundingBaseAttributes, GroundingDINO


class GroundingDINOClassificationAttributes(GroundingBaseAttributes):
    """Attributes for GroundingDINO Classification.

    Attributes:
        classes_file (str | None): Path to file containing classification classes.
        text_input (str | None): Input text for classification.
        top_k (int): The maximum number of top predictions to return. Defaults to 5.
            This specifies an upper limit, and fewer predictions may be returned
            if the number of valid predictions is less than `top_k`.
    """

    classes_file: str | None = None
    text_input: str | None = None
    top_k: int = 5


class GroundingDINOClassification(GroundingDINO):
    """GroundingDINO model for image classification tasks.

    This module provides an implementation of a zero-shot classification system
    using the GroundingDINO model. The classification pipeline allows for image
    classification tasks based on a predefined set of classes or text input.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: GroundingDINOClassification
      class_name: GroundingDINOClassification
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/model'
        model_cache_dir: /path/to/cache/dir
        inference_mode: 'object_detection'
        threshold: 0.25
        text_threshold: 0.25
        device: 'cuda'
        classes_file: /file/where/classes/are/stored
        text_input: object to detect
        top_k: 5

    """

    AttributesBaseModel = GroundingDINOClassificationAttributes

    def validate_and_format_text_input(self, text_input: str) -> str:
        """Validates and formats the text input for consistency.

        Args:
            text_input (str): The input text specifying object classes.

        Returns:
            str: A validated and formatted version of the input text.
        """
        if not self.attributes.classes_file:
            return super().validate_and_format_text_input(text_input)
        formatted_classes = self._process_classes()
        return super().validate_and_format_text_input(" ".join(formatted_classes))

    def _process_classes(self) -> list[str]:
        """Process classes from the classes file.

        Returns:
            list[str]: List of processed class names.
        """
        with open(self.attributes.classes_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return self._format_classes_with_limit(classes)

    def _format_classes_with_limit(self, classes: list[str]) -> list[str]:
        """Format classes while respecting token limit.

        Args:
            classes (list[str]): List of class names to format.

        Returns:
            list[str]: List of formatted class names within token limit.
        """
        formatted_classes = []
        total_tokens = 0
        for cls in classes:
            class_text = self._format_class_text(cls)
            tokens = len(self.processor.tokenizer.encode(class_text))
            if total_tokens + tokens >= self.max_tokens:
                break
            formatted_classes.append(class_text)
            total_tokens += tokens
        self._validate_classes(formatted_classes, total_tokens)
        return formatted_classes

    def _format_class_text(self, cls: str) -> str:
        """Format a class name with the appropriate delimiter.

        Args:
            cls (str): The class name to format.

        Returns:
            str: The formatted class name with the class delimiter appended if missing.
        """
        return f"{cls}{self.KEYS.CLASS_DELIMITER}" if not cls.endswith(self.KEYS.CLASS_DELIMITER) else cls

    def _validate_classes(self, classes: list[str], total_tokens: int) -> None:
        """Validate the list of classes and ensure they are within token limits.

        Args:
            classes (list[str]): The list of formatted class names.
            total_tokens (int): The total number of tokens used for the classes.

        Raises:
            ValueError: If no classes can be processed within the token limit.
        """
        if not classes:
            raise ValueError("No classes could be processed within token limit")
        log_msg = f"Processing {len(classes)} classes with {total_tokens} tokens"
        self.logger.debug(log_msg)

    def _create_annotations(self, image_packet: ImagePacket, results: list[dict[str, Any]]) -> None:
        """Creates annotation objects for the image packet based on filtered detection results.

        Args:
            image_packet (ImagePacket): Container holding the processed image and metadata.
            list[dict[str, Any]]: Detection results including bounding boxes, labels, and scores.
        """

        filtered_results = self._filter_results(results)
        annotations = self._create_annotations_from_results(filtered_results)
        image_packet.annotations.extend(annotations)

    def _create_annotations_from_results(self, results: list[dict[str, Any]]) -> list[ImageAnnotations]:
        """Create annotation objects from model results.

        Args:
            results (list[dict[str, Any]]): A list of results from the model inference.

        Returns:
            list[ImageAnnotations]: A list of annotation objects with labels, confidence scores,
            and other metadata.
        """
        return [
            ImageAnnotations(
                label=label,
                label_str=label,
                confidence_score=float(score),
                is_ground_truth=False,
            )
            for result in results
            for label, score in zip(result[self.KEYS.LABELS], result[self.KEYS.CONFIDENCE_SCORE])
        ]

    def _filter_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filters results to exclude annotations with empty or whitespace-only labels, and discards labels not found in
        the provided class names. Additionally, it sorts the received results and keep only the results with the top-k
        confidence scores.

        Args:
            results (list[dict[str, Any]]): List of detection results containing scores, labels,
                and boxes.

        Returns:
            list[dict[str, Any]]: A list of filtered top-k results.
        """

        class_names = self.get_class_names()
        best_predictions = self._get_best_predictions(results, class_names)
        return sorted(
            best_predictions.values(),
            key=lambda x: x[self.KEYS.CONFIDENCE_SCORE][0],
            reverse=True,
        )[: self.attributes.top_k]

    def _get_best_predictions(self, results: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
        """Get the best predictions for each class.

        Args:
            results (list[dict[str, Any]]): Model prediction results.
            class_names (list[str])): List of valid class names.

        Returns:
            dict[str, Any]: Dictionary of the best predictions per class.
        """
        best_predictions: dict = {}
        for result in results:
            for label, score in zip(result[self.KEYS.LABELS], result[self.KEYS.CONFIDENCE_SCORE]):
                self._update_prediction(label.strip(), float(score), class_names, best_predictions)
        return best_predictions

    def _update_prediction(self, label: str, score: float, class_names: list[str], predictions: dict[str, Any]) -> None:
        """Update the predictions dictionary with the best confidence score for a label.

        Args:
            label (str): The label associated with the current prediction.
            score (float): The confidence score of the current prediction.
            class_names (list[str]): A list of valid class names to consider for predictions.
            predictions (dict[str, Any]): A dictionary storing the best predictions per label. The keys
                are labels, and the values are dictionaries containing:
                - `self.KEYS.LABELS` (list[str]): List of label names.
                - `self.KEYS.CONFIDENCE_SCORE` (list[float]): List of confidence scores.
        """
        if label in class_names and (
            label not in predictions or score > predictions[label][self.KEYS.CONFIDENCE_SCORE][0]
        ):
            predictions[label] = {
                self.KEYS.LABELS: [label],
                self.KEYS.CONFIDENCE_SCORE: [score],
            }
