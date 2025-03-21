<h1 align="center">
<br>
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Hugging Face Embeddings
<br>
</h1>

<h4 align="center">Templates for seamless integration with Hugging Face embedding models</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">📦 Features</a> •
<a href="#example">📦 Example usage</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>


<h2 id="installation">🐍 Installation</h2>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-embeddings --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-embeddings --extra-index-url https://pypi.sinapsis.tech
```


Change the name of the package for the one you want to install.

> [!IMPORTANT]
> Templates in each package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>
with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-embeddings[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-embeddings[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">📦 Features</h2>


The templates in this package include multiple templates tailored for different **embedding-based** tasks:

- **SpeakerEmbeddingFromAudio**: Extracts speaker embeddings from **audio data** and attaches them to text or audio packets.
- **SpeakerEmbeddingFromDataset**: Retrieves speaker embeddings from **Hugging Face datasets** and integrates them into a DataContainer.
- **HuggingFaceEmbeddingNodeGenerator**: Generates **text embeddings**, splits documents into **chunks**, and processes them with metadata.

<h2 id="example">▶️ Example Usage</h2>

Below is an example YAML configuration for extracting **speaker embeddings** from an **audio file** and attaching them to **text packets**.

<details>
<summary ><strong><span style="font-size: 1.4em;">Config</span></strong></summary>

```yaml
agent:
  name: embeddings_agent

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: TextInput
    class_name: TextInput
    template_input: InputTemplate
    attributes:
      text: This is a test to check how the model works with a normal voice like mine.

  - template_name: AudioReaderSoundfile
    class_name: AudioReaderSoundfile
    template_input: TextInput
    attributes:
      audio_file_path: test.mp3

  - template_name: SpeakerEmbeddingFromAudio
    class_name: SpeakerEmbeddingFromAudio
    template_input: AudioReaderSoundfile
    attributes:
      target_packet: texts
```
</details>

> [!IMPORTANT]
> The TextInput and AudioReaderSoundfile templates correspond to the [sinapsis-data-readers](https://pypi.org/project/sinapsis-data-readers/) package. If you want to use the example, please make sure you install this package.
>

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




