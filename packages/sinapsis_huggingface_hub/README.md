<h1 align="center">
<br>
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Hugging Face Hub
<br>
</h1>

Sinapsis Hugging Face Hub provides a simple and flexible **no-code** implementation of the **Hugging Face Hub** library. It enables users to easily manage models, datasets, and spaces for Hugging Face-related tasks.

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">📦 Features</a> •
<a href="#example">▶️ Example usage</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>


<h2 id="installation">🐍 Installation</h2>


Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-hub --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-hub --extra-index-url https://pypi.sinapsis.tech
```



> [!IMPORTANT]
> Templates may require extra optional dependencies. For development, we recommend installing the package with all the optional dependencies:
>
with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-hub[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-hub[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">📦 Features</h2>

The templates in this package include functionality to:

- **HuggingFaceDownloader**: Downloads a repository snapshot from the Hugging Face Hub.

<h2 id="example">▶️ Example Usage</h2>

Below is an example YAML configuration for running a **Stable Diffusion Downloader** pipeline using Sinapsis.

<details>
<summary ><strong><span style="font-size: 1.4em;">Config</span></strong></summary>

```yaml
agent:
  name: stable_diffusion_agent_downloader

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: HuggingFaceDownloader
  class_name: HuggingFaceDownloader
  template_input: InputTemplate
  attributes:
    repo_id: stable-diffusion-v1-5/stable-diffusion-v1-5
    max_workers: 4
```
</details>

To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```

<h2 id="documentation">📙 Documentation</h2>

Documentation is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)

<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.




