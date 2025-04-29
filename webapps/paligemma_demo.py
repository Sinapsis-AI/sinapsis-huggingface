# -*- coding: utf-8 -*-
from typing import Any

import gradio as gr
import numpy as np
from diffusers_demo import needs_input_output
from sinapsis.webapp.agent_gradio_helper import (
    add_logo_and_title,
    css_header,
)
from sinapsis_core.cli.run_agent_from_config import generic_agent_builder
from sinapsis_core.data_containers.data_packet import (
    DataContainer,
    ImagePacket,
    TextPacket,
)
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP

CONFIGS_DIR = "packages/sinapsis_huggingface_transformers/src/sinapsis_huggingface_transformers/configs/"
CONFIG_NAME = "pali_gemma_detection.yml"
DEFAULT_CONF = CONFIGS_DIR + CONFIG_NAME
CONFIG_FILE = AGENT_CONFIG_PATH or DEFAULT_CONF


def build_demo(config_path: str = CONFIG_FILE) -> gr.Blocks:
    """Builds the Gradio interface for PaliGemma2 demo.

    Args:
        config_path (str): Path to the YAML configuration file. Defaults to CONFIG_FILE.

    Returns:
        gr.Blocks: A Gradio Blocks interface configured for image and text processing.
    """
    is_text_mode = needs_input_output(config_path, ("PaliGemmaInference",))
    agent = generic_agent_builder(config_path)

    def process(image: Any, text: str) -> np.ndarray | None:
        """Processes image and text input through the PaliGemma2 model.

        Args:
            image (np.ndarray): Input image array.
            text (str): Input prompt text.

        Returns:
            Union[str, PIL.Image.Image]: Either generated caption text or processed image
                depending on the configuration mode.

        Raises:
            gr.Warning: If image or text inputs are missing or invalid.
        """
        if image is None or not text:
            gr.Warning("Please provide both image and prompt")
            return None

        container = DataContainer()
        container.images = [ImagePacket(content=image, source="live_stream")]
        container.texts = [TextPacket(content=text, source="live_stream")]

        result = agent(container)

        if is_text_mode:
            return result.images[0].annotations[0].text
        return result.images[0].content

    with gr.Blocks(css=css_header()) as demo_interface:
        add_logo_and_title("Sinapsis PaliGemma2")

        gr.Interface(
            description="Using PaliGemma2 Vision-Language Model for zero-shot captioning, detection, and segmentation.",
            fn=process,
            inputs=[
                gr.Image(),
                gr.Textbox(label="Prompt"),
            ],
            outputs=(gr.Textbox(label="Generated Caption") if is_text_mode else gr.Image(type="pil")),
            article="Upload an image and provide a prompt to generate captions or visual analysis",
            flagging_mode="never",
        )

    return demo_interface


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=GRADIO_SHARE_APP)
