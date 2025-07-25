# -*- coding: utf-8 -*-
from enum import Enum


class Tags(Enum):
    BBOX = "bbox"
    DIFFUSERS = "diffusers"
    GENERATIVE = "generative"
    HUGGINGFACE = "huggingface"
    IMAGE = "image"
    IMAGE_TO_IMAGE = "image_to_image"
    IMAGE_TO_VIDEO = "image_to_video"
    IMAGE_GENERATION = "image_generation"
    INPAINTING = "inpainting"
    MASK = "mask"
    MODELS = "models"
    PROMPTS = "prompts"
    TEXT = "text"
    TEXT_TO_IMAGE = "text_to_image"
    VIDEO = "video"
    VIDEO_GENERATION = "video_generation"
