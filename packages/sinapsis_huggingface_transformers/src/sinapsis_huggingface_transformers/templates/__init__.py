# -*- coding: utf-8 -*-
import importlib
from typing import Callable

_root_lib_path = "sinapsis_huggingface_transformers.templates"

_template_lookup = {
    "ImageToTextTransformers": f"{_root_lib_path}.image_to_text_transformers",
    "PaliGemmaDetection": f"{_root_lib_path}.pali_gemma.pali_gemma_detection",
    "PaliGemmaInference": f"{_root_lib_path}.pali_gemma.pali_gemma_inference",
    "PaliGemmaSegmentation": f"{_root_lib_path}.pali_gemma.pali_gemma_segmentation",
    "SpeechToTextTransformers": f"{_root_lib_path}.speech_to_text_transformers",
    "SummarizationTransformers": f"{_root_lib_path}.summarization_transformers",
    "TextToSpeechTransformers": f"{_root_lib_path}.text_to_speech_transformers",
    "TranslationTransformers": f"{_root_lib_path}.translation_transformers",
}


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
