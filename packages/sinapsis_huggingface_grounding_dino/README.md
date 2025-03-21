<h1 align="center">
<br>
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Hugging Face Grounding DINO
<br>
</h1>

<h4 align="center">Templates for seamless integration with Grounding DINO models</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">📦 Features</a> •
<a href="#example">▶️ Example Usage</a> •
<a href="#webapp">🌐 Webapp</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>


<h2 id="installation">🐍 Installation</h2>


Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-grounding-dino --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-grounding-dino --extra-index-url https://pypi.sinapsis.tech
```



> [!IMPORTANT]
> Templates may require extra optional dependencies. For development, we recommend installing the package with all the optional dependencies:
>
with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-grounding-dino[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-grounding-dino[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">📦 Features</h2>

The templates in this package include functionality for different **Grounding DINO-based** tasks:

- **GroundingDINO**: Detect objects with **bounding boxes** based on text prompts (**zero-shot object detection**).
- **GroundingDINOClassification**: Classify images using **predefined classes or text prompts**, handling as many classes as possible within token limits.
- **GroundingDINOFineTuning**: Fine-tune **Grounding DINO checkpoints** on custom datasets.

<h2 id="example">▶️ Example Usage</h2>

Below is an example YAML configuration for **zero-shot object detection** using **Grounding DINO**.

<details>
<summary ><strong><span style="font-size: 1.4em;">Config</span></strong></summary>


```yaml
agent:
  name: grounding_dino_detection

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: FolderImageDatasetCV2
    class_name: FolderImageDatasetCV2
    template_input: InputTemplate
    attributes:
      data_dir: my_dataset

  - template_name: GroundingDINO
    class_name: GroundingDINO
    template_input: FolderImageDatasetCV2
    attributes:
      model_path: IDEA-Research/grounding-dino-base
      inference_mode: zero_shot
      text_input: a person.
      device: cuda
      threshold: 0.2
      text_threshold: 0.3

  - template_name: BBoxDrawer
    class_name: BBoxDrawer
    template_input: GroundingDINO
    attributes:
      overwrite: true
      randomized_color: false

  - template_name: ImageSaver
    class_name: ImageSaver
    template_input: BBoxDrawer
    attributes:
      save_dir: ./output_dir
      extension: png
```
</details>

> [!IMPORTANT]
> The FolderImageDatasetCV2, BBoxDrawer and ImageSaver templates correspond to the [sinapsis-data-readers](https://pypi.org/project/sinapsis-data-readers/), [sinapsis-data-visualization](https://pypi.org/project/sinapsis-data-visualization/) and [sinapsis-data-writers](https://pypi.org/project/sinapsis-data-writers/) packages respectively. If you want to use the example, please make sure you install these packages.
>

To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```

<h2 id="webapp">🌐 Webapp</h2>

Th **Sinapsis web applications** provide an interactive way to explore and experiment with AI models. They allow users to generate outputs, test different inputs, and visualize results in real time, making it easy to experience the capabilities of each model. Below are the available webapps and instructions to launch them.

> [!IMPORTANT]
> To run any of the apps, you first need to clone this repo:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-huggingface.git
cd sinapsis-huggingface
```


> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`.

> [!NOTE]
> Agent configuration can be changed through the AGENT_CONFIG_PATH env var. You can check the available configurations in each package configs folder.



<details>
<summary id="docker"><strong><span style="font-size: 1.4em;">🐳 Build with Docker</span></strong></summary>

**IMPORTANT** The docker image depends on the sinapsis-nvidia:base image. To build it, refer to the [official sinapsis documentation]([https://](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker)


1. **Build the sinapsis-huggingface image**:
```bash
docker compose -f docker/compose.yaml build
```
2. **Start the container**:
```bash
docker compose -f docker/compose_vision.yaml up sinapsis-huggingface-vision-gradio -d
```

3. **Check the status**:
```bash
docker logs -f sinapsis-huggingface-vision-gradio
```

4. **The logs will display the URL to access the webapp, e.g.,**:
```bash
Running on local URL:  http://127.0.0.1:7860
```
**NOTE**: The local URL can be different, please check the logs

5. **To stop the app**:
```bash
docker compose -f docker/compose_vision.yaml down
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
uv run webapps/vision_demo.py
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




