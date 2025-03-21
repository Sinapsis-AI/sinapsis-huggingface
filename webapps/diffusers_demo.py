# -*- coding: utf-8 -*-

import gradio as gr
from sinapsis.webapp.agent_gradio_helper import (
    add_logo_and_title,
    css_header,
    init_image_inference,
)
from sinapsis_core.agent.builder import load_yaml_config
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP

DEFAULT_CONFIG = "packages/sinapsis_huggingface_diffusers/src/sinapsis_huggingface_diffusers/configs/text_to_image.yml"
CONFIG_FILE = AGENT_CONFIG_PATH or DEFAULT_CONFIG


def needs_input_output(config_path: str, template_names: tuple[str, ...]) -> bool:
    """
    Determines if the diffusion pipeline requires an input image based on the configuration.

    Args:
        config_path (str): Path to the YAML configuration file.
        template_names (tuple[str]): Names of templates to check if are present in the configuration.

    Returns:
        bool: True if the model requires an input image (Image2Image or Inpainting),
              False otherwise.
    """
    config = load_yaml_config(config_path)
    return any(template["class_name"] in template_names for template in config["templates"])


def build_demo(config_path: str = CONFIG_FILE, title: str = "Sinapsis Diffusers Demo") -> gr.Interface:
    """
    Builds and configures the appropriate Gradio demo interface based on the model type.

    This function determines whether to create a text-to-image interface or an
    image-to-image/inpainting interface based on the configuration.

    Args:
        config_path (str): Path to the YAML configuration file. Defaults to CONFIG_FILE.
        title (str): Title for the Gradio interface. Defaults to "Sinapsis Diffusers Demo".

    Returns:
        gr.Interface: The configured Gradio interface.
    """
    image_input = gr.Image
    if not needs_input_output(config_path, ("ImageToImageDiffusers", "InpaintingDiffusers")):
        image_input = None

    with gr.Blocks(title=title, css=css_header()) as demo:
        add_logo_and_title(title)
        init_image_inference(config_path, image_input=image_input)
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=GRADIO_SHARE_APP)
