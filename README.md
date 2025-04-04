<h1 align="center">
<br>
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Hugging Face
<br>
</h1>

<h4 align="center">Package providing seamless integration with Hugging Face models, specializing in zero-shot object detection, classification, segmentation, generative workflows, and embeddings. It leverages state-of-the-art tools like Grounding DINO, Hugging Face Diffusers, and Transformers, enabling efficient implementation and customization.</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#packages">📦 Packages</a> •
<a href="#webapps">🌐 Webapps</a> •
<a href="#webapps">📙 Documentation</a> •
<a href="#packages">🔍 License</a>
</p>


<h2 id="installation">🐍 Installation</h2>

This repo consists of different packages to handle huggingface tools for different tasks:

* <code>sinapsis-huggingface-diffusers</code>
* <code>sinapsis-huggingface-embeddings</code>
* <code>sinapsis-huggingface-grounding-dino</code>
* <code>sinapsis-huggingface-transformers</code>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-diffusers --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-diffusers --extra-index-url https://pypi.sinapsis.tech
```


Change the name of the package for the one you want to install.

> [!IMPORTANT]
> Templates in each package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>
with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-diffusers[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-diffusers[all] --extra-index-url https://pypi.sinapsis.tech
```
Change the name of the package accordingly

> [!TIP]
> You can also install all the packages within this project:

```bash
  uv pip install sinapsis-huggingface[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="packages">📦 Packages</h2>

This repository is structured into modular packages, each designed for specific Hugging Face model integrations. These packages provide ready-to-use templates for tasks like text generation, embeddings, object detection, and diffusion-based image processing.

Each package can be used independently or combined to create more complex workflows. Below is an overview of the available packages:
<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">Sinapsis Hugging Face Diffusers</span></strong></summary>

This sinapsis package provides a powerful and flexible implementation of Hugging Face's diffusers library. It includes:

- Templates for tasks like **text-to-image**, **image-to-image**, **inpainting**, and **image-to-video generation**.
- Support for state-of-the-art models like **Stable Diffusion** and other diffusion-based architectures.
- Robust pipelines for generating and transforming visual content.

For specific instructions and further details, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-huggingface/blob/main/packages/sinapsis_huggingface_diffusers/README.md).
</details>
<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">Sinapsis Hugging Face Embeddings</span></strong></summary>

This package provides templates for generating and managing embeddings using Hugging Face models:

- **Speaker Embeddings**: Extract embeddings from audio packets or pre-defined Hugging Face datasets and attach them to audio or text packets.
- **Text Embeddings**: Generate embeddings for documents, with support for customizable chunking and metadata handling.

For more details, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-huggingface/blob/main/packages/sinapsis_huggingface_embeddings/README.md).

</details>
<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">Sinapsis Hugging Face Grounding DINO</span></strong></summary>

This sinapsis package provides **zero-shot detection and classification** capabilities using Hugging Face's Grounding DINO. It includes:

- Ready-to-use inference templates for object detection tasks and classification pipelines.
- Template for fine-tuning Grounding DINO checkpoints on specific datasets.

For detailed instructions and additional information, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-huggingface/blob/main/packages/sinapsis_huggingface_grounding_dino/README.md).


</details>
<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">Sinapsis Hugging Face Transformers</span></strong></summary>

This sinapsis package offers advanced capabilities for **text, speech, and image processing tasks**. It includes a variety of customizable inference templates designed for seamless integration into machine learning workflows:

- **Text-to-Speech (TTS) Template**: Convert text into high-quality, natural-sounding speech.
- **Speech-to-Text (STT) Template**: Transcribe spoken audio into text with support for multiple languages.
- **Translation Template**: Translate text from one language to another with support for various source and target languages.
- **Summarization Template**: Condense long-form content into concise summaries.
- **Image-to-Text Template**: Generate textual descriptions from input images.

For more details and specific templates, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-huggingface/blob/main/packages/sinapsis_huggingface_transformers/README.md).

</details>

For more details, see the [official documentation](https://docs.sinapsis.tech/docs)

<h2 id="webapps">🌐 Webapps</h2>

The **Sinapsis web applications** provide an interactive way to explore and experiment with AI models. They allow users to generate outputs, test different inputs, and visualize results in real time, making it easy to experience the capabilities of each model. Below are the available webapps and instructions to launch them.

> [!IMPORTANT]
> To run any of the apps, you first need to clone this repo:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-huggingface.git
cd sinapsis-huggingface
```

> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

> [!NOTE]
> Agent configuration can be changed through the AGENT_CONFIG_PATH env var. You can check the available configurations in each package configs folder.

> [!IMPORTANT]
> Please make sure you have a valid huggingface access token in order to run the paligemma webapp. For further instructions on how to create an access token see
https://huggingface.co/docs/transformers.js/en/guides/private




<details>
<summary id="docker"><strong><span style="font-size: 1.4em;">🐳 Build with Docker</span></strong></summary>

**IMPORTANT** The docker image depends on the sinapsis-nvidia:base image. To build it, refer to the [official sinapsis documentation](https://github.com/Sinapsis-AI/sinapsis/blob/main/README.md#docker)


1. **Build the sinapsis-huggingface image**:
```bash
docker compose -f docker/compose.yaml build
```
2. **Start the container**:

For Diffusers app
```bash
docker compose -f docker/compose_diffusers.yaml up sinapsis-huggingface-diffusers-gradio -d
```
For Grounding-Dino app
```bash
docker compose -f docker/compose_vision.yaml up sinapsis-huggingface-vision-gradio -d
```
For Paligemma app

```bash
export HF_TOKEN="your_huggingface_token"
docker compose -f docker/compose_pali_gemma.yaml up sinapsis-huggingface-paligemma-gradio -d
```
3. **Check the status**:

For Diffusers app
```bash
docker logs -f sinapsis-huggingface-diffusers-gradio
```
For Grounding-Dino app
```bash
docker logs -f sinapsis-huggingface-vision-gradio
```
For Paligemma app
```bash
docker logs -f sinapsis-huggingface-paligemma-gradio
```
**NOTE**: If using the vision app, please change the name of the service accordingly

4. **The logs will display the URL to access the webapp, e.g.,**:
```bash
Running on local URL:  http://127.0.0.1:7860
```
**NOTE**: The local URL can be different, please check the logs

5. **To stop the app**:

For Diffusers app
```bash
docker compose -f docker/compose_diffusers.yaml down
```
For Grounding-Dino app
```bash
docker compose -f docker/compose_vision.yaml down
```
For Paligemma app
```bash
docker compose -f docker/compose_pali_gemma.yaml down
```
</details>

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">📦 UV</span></strong></summary>

1. Create the virtual environment and sync the dependencies:

```bash
uv sync --frozen
```

2. Install the dependencies:

```bash
uv pip install sinapsis-huggingface[all] --extra-index-url https://pypi.sinapsis.tech
```
3. Run the webapp.

For Diffusers app
```bash
uv run webapps/diffusers_demo.py
```
For Grounding-Dino app
```bash
uv run webapps/vision_demo.py
```
For Paligemma app
```bash
export HF_TOKEN="your_huggingface_token"
uv run webapps/paligemma_demo.py
```

4. The terminal will display the URL to access the webapp, e.g., :
```bash
Running on local URL:  http://127.0.0.1:7860
```

</details>


<h2 id="documentation">📙 Documentation</h2>

Documentation is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)

<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.




