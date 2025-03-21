# -*- coding: utf-8 -*-
import importlib
from typing import Any, Callable, cast

_root_lib_path = "sinapsis_huggingface_diffusers.templates"

_template_lookup = {
    "TextToImageDiffusers": f"{_root_lib_path}.text_to_image_diffusers",
    "ImageToImageDiffusers": f"{_root_lib_path}.image_to_image_diffusers",
    "ImageToVideoGenXLDiffusers": f"{_root_lib_path}.image_to_video_gen_xl_diffusers",
    "InpaintingDiffusers": f"{_root_lib_path}.inpainting_diffusers",
}


def __getattr__(name: str) -> Callable[..., Any]:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        attr = getattr(module, name)
        if callable(attr):
            return cast(Callable[..., Any], attr)
        raise TypeError(f"Attribute `{name}` in `{_template_lookup[name]}` is not callable.")

    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
