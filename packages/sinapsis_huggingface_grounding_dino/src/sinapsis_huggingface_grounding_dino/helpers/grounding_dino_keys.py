# -*- coding: utf-8 -*-
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class GroundingDINOKeys:
    """Defines key constants used in the GroundingDINO workflow for consistent referencing.

    Attributes:
        CLASS_DELIMITER (str): Delimiter used to separate class names in the input text.
        CONFIDENCE_SCORE (str): Key for accessing confidence scores from the model output.
        INPUT_IDS (str): Key for tokenized input IDs used by the model.
        LABELS (str): Key for general label data in the processed output.
        BBOXES (str): Key for bounding box coordinates in the output.
    """

    CLASS_DELIMITER: str = "."
    CONFIDENCE_SCORE: str = "scores"
    INPUT_IDS: str = "input_ids"
    LABELS: str = "text_labels"
    BBOXES: str = "boxes"
