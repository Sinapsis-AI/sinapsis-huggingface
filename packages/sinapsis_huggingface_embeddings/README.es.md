<h1 align="center"><br/><br/><a href="https://sinapsis.tech/"><img alt="" src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true" width="300"/></a><br/>Sinapsis Hugging Face Embeddings
<br/></h1>
<h4 align="center">Plantillas para la integración perfecta con modelos de embedding Hugging Face</h4>
<p align="center"><a href="#installation">🐍 Instalación</a> •
<a href="#features">📦 Características</a> •
<a href="#example">📦 Uso del ejemplo</a> •
<a href="#documentation">📙 Documentación</a> •
<a href="#license">🔍 Licencia</a>

<h2 id="installation">🐍 Instalación</h2>

Instala el administrador de paquetes de tu elección. Alentamos el uso de <code>uv</code>


Ejemplo con <code>uv</code>:



```bash
  uv pip install sinapsis-huggingface-embeddings --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
  pip install sinapsis-huggingface-embeddings --extra-index-url https://pypi.sinapsis.tech
```

Cambia el nombre del paquete para el que desea instalar.

<blockquote>

[!IMPORTANT]
Las plantillas en cada paquete pueden requerir dependencias adicionales. Para el desarrollo, recomendamos instalar el paquete con todas las dependencias opcionales:

con <code>uv</code>:

</blockquote>

```bash
  uv pip install sinapsis-huggingface-embeddings[all] --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
  pip install sinapsis-huggingface-embeddings[all] --extra-index-url https://pypi.sinapsis.tech
```
<h2 id="features">📦 Características</h2>
Las plantillas en este paquete incluyen múltiples plantillas adaptadas para diferentes <strong>incrustación basada en</strong> tareas:
<ul>
<li><strong>Presidente De Audio</strong>: Extractos de las incrustaciones de altavoces de <strong>Datos de audio</strong> y los adjunta a paquetes de texto o audio.</li>

<li><strong>SpeakerEmbeddingDesdeDataset</strong>: Recupera incrustaciones de habla con <strong>Hugging Face datasets</strong> y los integra en un DataContainer.</li>

<li><strong>HuggingFaceEmbeddingNodeGenerator</strong>: Genera <strong>texto incrustaciones</strong>, divide documentos en <strong>pedazos</strong>, y los procesa con metadatos.</li>
</ul><h2 id="example">Ejemplo de uso</h2>
A continuación se muestra un ejemplo de configuración YAML para la extracción de<strong> incrust aciones de habla</strong> de un <strong>Archivo de audio</strong> y adjuntarlos a <strong>paquetes de texto</strong>.


<details><summary><strong><span style="font-size: 1.4em;">Config</span></strong></summary>
</details>


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



<blockquote>

[!IMPORTANT]
Las plantillas TextInput y AudioReaderSoundfile corresponden al paquete <a href="https://pypi.org/project/sinapsis-data-readers/">sinapsis-data-readers</a>. Si desea utilizar el ejemplo, asegúrese de instalar este paquete.

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

