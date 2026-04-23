<h1 align="center"><br/><br/><a href="https://sinapsis.tech/"><img alt="" src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true" width="300"/></a><br/>Sinapsis Hugging Face Diffusers
<br/></h1>

Sinapsis Hugging Face Diffusers proporciona un potente y flexible <strong></strong> aplicación sin código de la biblioteca <strong> Hugging Face Diffusers</strong>. Permite a los usuarios configurar y ejecutar fácilmente <strong>tuberías de difusión</strong> para tareas generativas.

<p align="center"><a href="#installation">🐍 Instalación</a> •
<a href="#features">📦 Características</a> •
<a href="#example">↑ Uso del ejemplo</a> •
<a href="#webapps">🌐 Aplicaciones web</a> •
<a href="#documentation">📙 Documentación</a> •
<a href="#license">🔍 Licencia</a>

<h2 id="installation">🐍 Instalación</h2>

Instala el administrador de paquetes de tu elección. Alentamos el uso de <code>uv</code>


Ejemplo con <code>uv</code>:



```bash
  uv pip install sinapsis-huggingface-diffusers --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
  pip install sinapsis-huggingface-diffusers --extra-index-url https://pypi.sinapsis.tech
```

<blockquote>

[!IMPORTANT]
Las plantillas pueden requerir dependencias opcionales adicionales. Para el desarrollo, recomendamos instalar el paquete con todas las dependencias opcionales:

con <code>uv</code>:

</blockquote>

```bash
  uv pip install sinapsis-huggingface-diffusers[all] --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
  pip install sinapsis-huggingface-diffusers[all] --extra-index-url https://pypi.sinapsis.tech
```
<h2 id="features">📦 Características</h2>
Las plantillas en este paquete incluyen funcionalidad para:
<ul>
<li><strong>TextToImageDiffusers</strong>: Genera imágenes de los avisos de texto.</li>

<li><strong>ImageToImage Diffusers</strong>: Modifica las imágenes mediante transformaciones guiadas por texto.</li>

<li><strong>InpaintingDiffusers</strong>: Admite la edición selectiva de imágenes usando máscaras o cajas de fijación.</li>

<li><strong>ImageToVideoGenXLDiffusers</strong>: Convierte imágenes en vídeos usando <strong>I2VGen-XL</strong> modelo.</li>
</ul><h2 id="example">Ejemplo de uso</h2>
A continuación se muestra un ejemplo de configuración YAML para ejecutar un <strong>Difusión de texto a imagen</strong> tubería con Sinapsis.


<details><summary><strong><span style="font-size: 1.4em;">Config</span></strong></summary>
</details>


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



<blockquote>

[!IMPORTANT]
La plantilla ImageSaver corresponde a la <a href="https://pypi.org/project/sinapsis-data-writers/">sinapsis-data-writers</a> paquete. Si desea utilizar el ejemplo, asegúrese de instalar estos paquetes.

</blockquote>

Para ejecutar la configuración, utilice el CLI:

```bash
sinapsis run name_of_config.yml
```
<h2 id="webapps">🌐 Aplicaciones web</h2>
Th <strong>Aplicaciones web de Sinapsis</strong> proporcionar una manera interactiva de explorar y experimentar con los modelos AI. Permiten a los usuarios generar salidas, probar diferentes entradas y visualizar resultados en tiempo real, facilitando la experiencia de las capacidades de cada modelo. A continuación se presentan las aplicaciones web disponibles e instrucciones para lanzarlas.

<blockquote>

[!IMPORTANT]
Para ejecutar cualquiera de las aplicaciones, primero necesitas clonar este repo:

</blockquote>

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-huggingface.git
cd sinapsis-huggingface
```

<blockquote>

[!NOTE]
Si desea habilitar el intercambio de aplicaciones externas en Gradio, <code>export GRADIO_SHARE_APP=True</code>

[!NOTE]
La configuración del agente puede cambiarse a través de la <code>AGENT_CONFIG_PATH</code> env var. Puede comprobar las configuraciones disponibles en cada carpeta de configuración de paquetes.

</blockquote>


<details><summary id="docker"><strong><span style="font-size: 1.4em;">Construir con Docker</span></strong></summary>
</details>


<strong>IMPORTANTE</strong> La imagen del docker depende de la sinapsis-nvidia: imagen básica. Para construirlo, consulta el <a href="[https://](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker">documentación oficial de sinapsis</a>
<ol>
<li><strong>Construir la imagen sinapsis-huggingface</strong>:</li>
</ol>

```bash
docker compose -f docker/compose.yaml build
```
<ol start="2">
<li><strong>Iniciar el contenedor</strong>:</li>
</ol>

```bash
docker compose -f docker/compose_diffusers.yaml up sinapsis-huggingface-diffusers-gradio -d
```
<ol start="3">
<li><strong>Compruebe el estado</strong>:</li>
</ol>

```bash
docker logs -f sinapsis-huggingface-diffusers-gradio
```
<ol start="4">
<li><strong>Los registros mostrarán la URL para acceder a la aplicación web, por ejemplo,</strong>:</li>
</ol>

```bash
Running on local URL:  http://127.0.0.1:7860
```

<strong>NOTA</strong>: La URL local puede ser diferente, por favor revise los registros
<ol start="5">
<li><strong>Para detener la aplicación</strong>:</li>
</ol>

```bash
docker compose -f docker/compose_diffusers.yaml down
```




<details><summary id="uv"><strong><span style="font-size: 1.4em;">📦 UV</span></strong></summary>
</details>

<ol>
<li>Crear el entorno virtual y sincronizar las dependencias:</li>
</ol>

```bash
uv sync --frozen
```
<ol start="2">
<li>Instalar las dependencias:</li>
</ol>

```bash
uv pip install sinapsis-huggingface[all] --extra-index-url https://pypi.sinapsis.tech
```
<ol start="3">
<li>Corre la aplicación web.</li>
</ol>

```bash
uv run webapps/diffusers_demo.py
```
<ol start="4">
<li>El terminal mostrará la URL para acceder a la aplicación web, por ejemplo:</li>
</ol>

```bash
Running on local URL:  http://127.0.0.1:7860
```


<h2 id="documentation">📙 Documentación</h2>
La documentación está disponible <a href="https://docs.sinapsis.tech/docs">web de sinapsis</a>

Los tutoriales para diferentes proyectos dentro de sinapsis están disponibles en <a href="https://docs.sinapsis.tech/tutorials">sinapsis tutoriales página</a>
<h2 id="license">🔍 Licencia</h2>
Este proyecto está licenciado bajo la licencia AGPLv3, que fomenta la colaboración abierta y el intercambio. Para más detalles, consulta el archivo <a href="LICENSE">LICENSE</a>.

Para uso comercial, consulta el  <a href="https://sinapsis.tech"> sitio web oficial de Sinapsis</a> para información sobre la obtención de una licencia comercial.

