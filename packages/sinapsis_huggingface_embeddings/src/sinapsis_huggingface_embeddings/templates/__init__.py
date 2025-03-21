# -*- coding: utf-8 -*-
import importlib
from typing import Callable

_root_lib_path = "sinapsis_huggingface_embeddings.templates"

_template_lookup = {
    "HuggingFaceEmbeddingNodeGenerator": f"{_root_lib_path}.embedding_node_generator",
    "HuggingFaceEmbeddingExtractor": f"{_root_lib_path}.hugging_face_embedding_extractor",
    "SpeakerEmbeddingFromAudio": f"{_root_lib_path}.speaker_embedding_from_audio",
    "SpeakerEmbeddingFromDataset": f"{_root_lib_path}.speaker_embedding_from_dataset",
}


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
