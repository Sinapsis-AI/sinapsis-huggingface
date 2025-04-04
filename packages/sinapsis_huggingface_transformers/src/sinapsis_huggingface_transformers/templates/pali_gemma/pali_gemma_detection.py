# -*- coding: utf-8 -*-
from dataclasses import dataclass

from sinapsis_core.data_containers.annotations import ImageAnnotations
from sinapsis_core.template_base import TemplateAttributeType
from sinapsis_huggingface_transformers.templates.pali_gemma.pali_gemma_inference import (
    PaliGemmaInference,
    PaliGemmaInferenceAttributes,
)
from sinapsis_huggingface_transformers.thirdparty.helpers import (
    get_matches,
    parse_label,
    parse_location_tokens,
)


@dataclass(frozen=True)
class PaliGemmaDetectionKeys:
    "Keys to use during detection"

    detection_prompt: str = "detect {}"


class PaliGemmaDetectionAttributes(PaliGemmaInferenceAttributes):
    """Configuration attributes for PaliGemma object detection tasks.

    This class extends the base inference attributes to handle object detection specific configurations.

    Attributes:
        objects_to_detect (str | list[str]): Target objects to detect, can be a single string or list of strings
    """

    objects_to_detect: str | list[str]


class PaliGemmaDetection(PaliGemmaInference):
    """Implementation of PaliGemma object detection pipeline.

    The template inherits functionality from its base class, extending
    the functionality to run inference on an image and to identify
    the objects from the attributes.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: PaliGemmaDetection
      class_name: PaliGemmaDetection
      template_input: InputTemplate
      attributes:
        model_path: '/path/to/paligemma/model'
        processor_path: '`/path/to/processor'
        model_cache_dir: /path/to/cache/dir
        device: 'cuda'
        max_new_tokens: 200
        torch_dtype: float16
        prompt: <image> caption en
        objects_to_detect: 'object to detect'

    """

    AttributesBaseModel = PaliGemmaDetectionAttributes
    KEYS = PaliGemmaDetectionKeys

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)

        objects_str = self.initialize_objects_str()
        self.prompt = self.KEYS.detection_prompt.format(objects_str)

    def initialize_objects_str(self) -> str:
        """
        Initialize the objects to detect string according to the specified format.

        Returns:
            str: String enlisting the objects to be defined in the detection prompt.
        """

        if isinstance(self.attributes.objects_to_detect, str):
            return self.attributes.objects_to_detect
        return "; ".join(self.attributes.objects_to_detect)

    def _format_text_for_prompt(self, text: str) -> str:
        """Formats input text as a detection prompt.

        Args:
            text (str): Raw text content (expected to be objects to detect)

        Returns:
            str: Formatted detection prompt
        """
        return self.KEYS.detection_prompt.format(text)

    def _create_annotation(
        self, caption: str, confidence: float, image_shape: tuple[int, ...]
    ) -> list[ImageAnnotations]:
        """Creates structured annotations from detection model outputs.

        Processes the model's output caption to extract bounding box coordinates
        and object labels for each detected instance.

        Args:
            caption (str): Raw detection output from the model
            confidence (float): Confidence score for the predictions
            image_shape (tuple[int, ...]): Dimensions of the input image (height, width)

        Returns:
            list[ImageAnnotations]: List of annotations containing bounding boxes and labels
                                  for each detected object
        """
        annotations = []
        matches = get_matches(caption)

        for match_coord in matches:
            coords = parse_location_tokens(match_coord, image_shape)
            label = parse_label(match_coord)
            annotation = self.create_bbox_annotation(coords, label, confidence)
            annotations.append(annotation)

        return annotations
