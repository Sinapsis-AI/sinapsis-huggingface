</h1>
<h1 align="center">
<br>
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Hugging Face Transformers
<br>
</h1>

<h4 align="center">Templates for seamless integration with Transformers models</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">📂 Features</a> •
<a href="#example">▶️ Example usage</a> •
<a href="#documentation">📦 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

<h2 id="installation">🐍 Installation</h2>


Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-transformers --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-transformers --extra-index-url https://pypi.sinapsis.tech
```
> [!IMPORTANT]
> Templates may require extra optional dependencies. For development, we recommend installing the package with all the optional dependencies:
>
with <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-transformers[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-huggingface-transformers[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">📦 Features</h2>

Sinapsis Hugging Face Transformers provides customizable inference templates for a variety of tasks, including **image captioning**, **object detection**, **instance segmentation**, **speech-to-text**, and **text-to-speech**.

**Templates:**


- **ImageToTextTransformers**: Generates textual descriptions from input images using Hugging Face image-to-text models.
- **PaliGemmaInference**: Generate captions for images.
- **PaliGemmaDetection**: Detect specific objects in images.
- **SpeechToTextTransformers**: Converts spoken audio into text using automatic speech recognition (ASR) models.
- **SummarizationTransformers**: Summarizes long text into concise summaries using Hugging Face summarization models.
- **TextToSpeechTransformers**: Converts text into lifelike audio using text-to-speech (TTS) models.
- **TranslationTransformers**: Translates text from a source language to a target language using Hugging Face translation models.




<h2 id="example">▶️ Example Usage</h2>

Below is an example YAML configuration for **text-to-speech (TTS) conversion** using the **Suno Bark** model.

<details>
<summary ><strong><span style="font-size: 1.4em;">Config</span></strong></summary>

```yaml
agent:
  name: test_agent

templates:
  - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}

  - template_name: TextInput
    class_name: TextInput
    attributes:
      text: Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe.

  - template_name: TextToSpeechTransformers
    class_name: TextToSpeechTransformers
    template_input: TextInput
    attributes:
      model_path: 'suno/bark'
      device: "cuda"
      torch_dtype: float32
      seed: 7
      use_embeddings: false
      n_words: 30
      inference_kwargs:
        generate_kwargs:
          do_sample: true
          temperature: 0.7

  - template_name: AudioWriterSoundfile
    class_name: AudioWriterSoundfile
    template_input: TextToSpeechTransformers
    attributes:
      root_dir: ./test
      save_dir: audios
```
</details>

> [!IMPORTANT]
> The TextInput and AudioWriterSoundfile templates correspond to the [sinapsis-data-readers](https://pypi.org/project/sinapsis-data-readers/) and [sinapsis-data-writers](https://pypi.org/project/sinapsis-data-writers/) packages respectively. If you want to use the example, please make sure you install these packages.
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