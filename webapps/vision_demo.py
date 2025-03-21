# -*- coding: utf-8 -*-
import os

import gradio as gr
from sinapsis.webapp.agent_gradio_helper import (
    add_logo_and_title,
    css_header,
    init_image_inference,
)
from sinapsis_core.utils.env_var_keys import GRADIO_SHARE_APP

IS_DOCKER = os.getenv("IS_DOCKER", "false").lower() == "true"
ROOT_CONFIG_PATH = (
    "/app/sinapsis_huggingface/configs/"
    if IS_DOCKER
    else "packages/sinapsis_huggingface_grounding_dino/src/sinapsis_huggingface_grounding_dino/configs/"
)

CONFIG_PATHS = {
    "Detection": os.getenv(
        "DETECTION_CONFIG_PATH",
        ROOT_CONFIG_PATH + "grounding_dino_detection.yml",
    ),
    "Segmentation": os.getenv(
        "SEGMENTATION_CONFIG_PATH",
        ROOT_CONFIG_PATH + "grounding_dino_ultralytics_segmentation.yml",
    ),
    "Classification": os.getenv(
        "CLASSIFICATION_CONFIG_PATH",
        ROOT_CONFIG_PATH + "grounding_dino_classification.yml",
    ),
}


def create_demo() -> gr.Blocks:
    """Creates and returns the Gradio Blocks demo interface.

    Returns:
        gr.Blocks: The Gradio Blocks object containing the entire interface.
    """
    with gr.Blocks(title="Sinapsis Zero-Shot Vision Models", css=css_header()) as demo:
        add_logo_and_title("Sinapsis Zero-Shot Vision Models")
        gr.Markdown("Explore zero-shot object detection, segmentation, and classification with Grounding DINO.")
        for task, config in CONFIG_PATHS.items():
            with gr.Tab(task):
                init_image_inference(config)
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=GRADIO_SHARE_APP)
