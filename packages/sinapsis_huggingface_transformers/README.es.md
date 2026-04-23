


<h1 align="center"><br/><br/><a href="https://sinapsis.tech/"><img alt="" src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true" width="300"/></a><br/>Sinapsis Hugging Face Transformers
<br/></h1>
<h4 align="center">Plantillas para la integración perfecta con los modelos Transformers</h4>
<p align="center"><a href="#installation">🐍 Instalación</a> •
<a href="#features">📂 Características</a> •
<a href="#example">↑ Uso del ejemplo</a> •
<a href="#documentation">📦 Documentación</a> •
<a href="#license">🔍 Licencia</a>


<h2 id="installation">🐍 Instalación</h2>
Instala el administrador de paquetes de tu elección. Alentamos el uso de <code>uv</code>

Ejemplo con <code>uv</code>:

```bash
  uv pip install sinapsis-huggingface-transformers --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
  pip install sinapsis-huggingface-transformers --extra-index-url https://pypi.sinapsis.tech
```

<blockquote>

[!IMPORTANT]
Las plantillas pueden requerir dependencias opcionales adicionales. Para el desarrollo, recomendamos instalar el paquete con todas las dependencias opcionales:

con <code>uv</code>:

</blockquote>

```bash
  uv pip install sinapsis-huggingface-transformers[all] --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
  pip install sinapsis-huggingface-transformers[all] --extra-index-url https://pypi.sinapsis.tech
```
<h2 id="features">📦 Características</h2>
Sinapsis Hugging Face Transformers proporciona plantillas de inferencia personalizables para una variedad de tareas, incluyendo <strong>imagen capción</strong>, <strong>detección de objetos</strong>, <strong>instancia segmentación</strong>, <strong>discurso a texto</strong>, y <strong>texto a palabra</strong>.

<strong>Plantillas:</strong>
<ul>
<li><strong>ImageToTextTransformers</strong>: Genera descripciones textuales de imágenes de entrada utilizando modelos de imagen a texto Hugging Face.</li>

<li><strong>PaliGemmaInference</strong>: Generar capciones para imágenes.</li>

<li><strong>PaliGemmaDetección</strong>: Detectar objetos específicos en imágenes.</li>

<li><strong>SpeechToTextTransformers</strong>: Convierte el audio hablado en texto usando modelos automáticos de reconocimiento del habla (ASR).</li>

<li><strong>Summarization Transformadores</strong>: Summarizes texto largo en resúmenes concisos usando modelos de resumen Hugging Face.</li>

<li><strong>TextToSpeechTransformers</strong>: Convierte texto en audio de estilo vivo usando modelos de texto a voz (TTS).</li>

<li><strong>TraducciónTransformers</strong>: Traduce texto de un idioma fuente a un idioma objetivo usando modelos de traducción Hugging Face.</li>
</ul><h2 id="example">Ejemplo de uso</h2>
A continuación se muestra un ejemplo de configuración YAML para <strong>conversión de texto a palabra (TTS)</strong> usando el <strong>Suno Bark</strong> modelo.


<details><summary><strong><span style="font-size: 1.4em;">Config</span></strong></summary>
</details>


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



<blockquote>

[!IMPORTANT]
Las plantillas TextInput y AudioWriterSoundfile corresponden a las <a href="https://pypi.org/project/sinapsis-data-readers/">sinapsis-data-readers</a> y <a href="https://pypi.org/project/sinapsis-data-writers/">sinapsis-data-writers</a> paquetes, respectivamente. Si desea utilizar el ejemplo, asegúrese de instalar estos paquetes.

</blockquote>

Para ejecutar la configuración, utilice el CLI:

```bash
sinapsis run name_of_config.yml
```
<h2 id="documentation">📙 Documentación</h2>
La documentación está disponible <a href="https://docs.sinapsis.tech/docs">web de sinapsis</a>

Los tutoriales para diferentes proyectos dentro de sinapsis están disponibles en <a href="https://docs.sinapsis.tech/tutorials">sinapsis tutoriales página</a>
<h2 id="license">🔍 Licencia</h2>
Este proyecto está licenciado bajo la licencia AGPLv3, que fomenta la colaboración abierta y el intercambio. Para más detalles, consulta el archivo <a href="LICENSE">LICENSE</a>.

Para uso comercial, consulta el  <a href="https://sinapsis.tech"> sitio web oficial de Sinapsis</a> para información sobre la obtención de una licencia comercial.

