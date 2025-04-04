# -*- coding: utf-8 -*-
"""
The constants and methods declared in this file are inspired in the following source:

https://github.com/google/generative-ai-docs/blob/main/site/en/gemma/docs/paligemma/inference-with-keras.ipynb

which is Licensed under the Apache License, Version 2.0.

"""

import numpy as np
import regex as re

COORDS_PATTERN: str = r"<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>"
LABEL_PATTERN: str = r" (?P<label>.+?)( ;|$)"

DETECTION_PATTERN: str = COORDS_PATTERN + LABEL_PATTERN

LOCATION_KEYS: tuple[str, ...] = ("y0", "x0", "y1", "x1")
LOCATION_SCALE: float = 1024.0


def parse_location_tokens(match_coord: re.Match, image_shape: tuple[int, ...]) -> np.ndarray:
    """Parses location tokens from model output into normalized coordinates.

    Args:
        match_coord (dict): Dictionary containing matched location tokens
        image_shape (tuple[int, ...]): Shape of the input image

    Returns:
        np.ndarray: Normalized coordinates (x0, y0, x1, y1)
    """
    match_dict = match_coord.groupdict()
    x0 = float(match_dict[LOCATION_KEYS[1]]) / LOCATION_SCALE * image_shape[1]
    y0 = float(match_dict[LOCATION_KEYS[0]]) / LOCATION_SCALE * image_shape[0]
    x1 = float(match_dict[LOCATION_KEYS[3]]) / LOCATION_SCALE * image_shape[1]
    y1 = float(match_dict[LOCATION_KEYS[2]]) / LOCATION_SCALE * image_shape[0]
    return np.array([x0, y0, x1, y1])


def parse_label(match_coord: re.Match) -> str:
    """
    Retrieves detection label from a regex Match object.


    Args:
        match_coord (Match): The Match object containing the label information.

    Returns:
        str: The detection label.
    """
    label = match_coord.groupdict().get("label")
    if label is None:
        return ""
    return label.strip()


def get_matches(caption: str) -> re.Scanner:
    """
    Creates an iterable containing all the detection matches found in the
    produced model caption.

    Args:
        caption (str): The caption produced by the paligemma model.

    Returns:
        Scanner: An iterable object containing all the regex matches.
    """

    return re.finditer(DETECTION_PATTERN, caption)
