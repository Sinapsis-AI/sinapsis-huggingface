<h1 align="center">
<br>
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Hugging Face Diffusers
<br>
</h1>

Sinapsis Hugging Face Diffusers provides a powerful and flexible **no-code** implementation of the **Hugging Face Diffusers** library. It enables users to easily configure and run **diffusion pipelines** for generative tasks.

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">📦 Features</a> •
<a href="#example">▶️ Example usage</a> •
<a href="#webapps">🌐 Webapps</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>


<h2 id="installation">🐍 Installation</h2>


Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-diffusers --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-diffusers --extra-index-url https://pypi.sinapsis.tech
```



> [!IMPORTANT]
> Templates may require extra optional dependencies. For development, we recommend installing the package with all the optional dependencies:
>
with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-diffusers[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-diffusers[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">📦 Features</h2>

The templates in this package include functionality to:

- **TextToImageDiffusers**: Generates images from text prompts.
- **ImageToImageDiffusers**: Modifies images using text-guided transformations.
- **InpaintingDiffusers**: Supports selective image editing using masks or bounding boxes.
- **ImageToVideoGenXLDiffusers**: Converts images into videos using the **I2VGen-XL** model.

<h2 id="example">▶️ Example Usage</h2>

Below is an example YAML configuration for running a **Text-to-Image Diffusion** pipeline using Sinapsis.

<details>
<summary ><strong><span style="font-size: 1.4em;">Config</span></strong></summary>

```yaml
agent:
  name: text_to_image

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: TextToImageDiffusers
    class_name: TextToImageDiffusers
    template_input: InputTemplate
    attributes:
      model_path: stable-diffusion-v1-5/stable-diffusion-v1-5
      device: cuda
      torch_dtype: "float16"
      enable_model_cpu_offload: false
      overwrite_images: true
      generation_params:
        prompt: "A majestic castle on top of a mountain, surrounded by clouds during sunset"
        height: 1024
        width: 1024
        num_inference_steps: 50
        guidance_scale: 7.5
        negative_prompt: "low quality, blurry, distorted"
        num_images_per_prompt: 1

  - template_name: ImageSaver
    class_name: ImageSaver
    template_input: TextToImageDiffusers
    attributes:
      save_dir: ./output_dir
      extension: png
```
</details>

> [!IMPORTANT]
> The ImageSaver template correspond to the [sinapsis-data-writers](https://pypi.org/project/sinapsis-data-writers/) package. If you want to use the example, please make sure you install this packages.
>

To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```


<h2 id="webapps">🌐 Webapps</h2>

Th **Sinapsis web applications** provide an interactive way to explore and experiment with AI models. They allow users to generate outputs, test different inputs, and visualize results in real time, making it easy to experience the capabilities of each model. Below are the available webapps and instructions to launch them.

> [!IMPORTANT]
> To run any of the apps, you first need to clone this repo:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-huggingface.git
cd sinapsis-huggingface
```


> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

> [!NOTE]
> Agent configuration can be changed through the `AGENT_CONFIG_PATH` env var. You can check the available configurations in each package configs folder.



<details>
<summary id="docker"><strong><span style="font-size: 1.4em;">🐳 Build with Docker</span></strong></summary>

**IMPORTANT** The docker image depends on the sinapsis-nvidia:base image. To build it, refer to the [official sinapsis documentation]([https://](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker)


1. **Build the sinapsis-huggingface image**:
```bash
docker compose -f docker/compose.yaml build
```
2. **Start the container**:
```bash
docker compose -f docker/compose_diffusers.yaml up sinapsis-huggingface-diffusers-gradio -d
```

3. **Check the status**:
```bash
docker logs -f sinapsis-huggingface-diffusers-gradio
```

4. **The logs will display the URL to access the webapp, e.g.,**:
```bash
Running on local URL:  http://127.0.0.1:7860
```
**NOTE**: The local URL can be different, please check the logs

5. **To stop the app**:
```bash
docker compose -f docker/compose_diffusers.yaml down
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

```bash
uv run webapps/diffusers_demo.py
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




