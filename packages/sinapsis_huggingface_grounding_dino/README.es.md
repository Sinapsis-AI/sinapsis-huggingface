<h1 align="center"><br/><br/><a href="https://sinapsis.tech/"><img alt="" src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true" width="300"/></a><br/>Sinapsis Hugging Face Grounding DINO
<br/></h1>
<h4 align="center">Plantillas para la integración sin costuras con modelos Grounding DINO</h4>
<p align="center"><a href="#installation">🐍 Instalación</a> •
<a href="#features">📦 Características</a> •
<a href="#example">Ejemplo de uso</a> •
<a href="#webapp">🌐 Aplicación web</a> •
<a href="#documentation">📙 Documentación</a> •
<a href="#license">🔍 Licencia</a>

<h2 id="installation">🐍 Instalación</h2>

Instala el administrador de paquetes de tu elección. Alentamos el uso de <code>uv</code>


Ejemplo con <code>uv</code>:



```bash
  uv pip install sinapsis-huggingface-grounding-dino --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
  pip install sinapsis-huggingface-grounding-dino --extra-index-url https://pypi.sinapsis.tech
```

<blockquote>

[!IMPORTANT]
Las plantillas pueden requerir dependencias opcionales adicionales. Para el desarrollo, recomendamos instalar el paquete con todas las dependencias opcionales:

con <code>uv</code>:

</blockquote>

```bash
  uv pip install sinapsis-huggingface-grounding-dino[all] --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
  pip install sinapsis-huggingface-grounding-dino[all] --extra-index-url https://pypi.sinapsis.tech
```
<h2 id="features">📦 Características</h2>
Las plantillas en este paquete incluyen funcionalidad para diferentes <strong>Basado en DINO</strong> tareas:
<ul>
<li><strong>GroundingDINO</strong>: Detectar objetos con <strong>Cajas de fijación</strong> basado en los prompts de texto (<strong>Detección de objetos en cero</strong>).</li>

<li><strong>GroundingDINOClassification</strong>: Clasificar imágenes utilizando <strong>clases predefinidas o instrucciones de texto</strong>, manejando tantas clases como sea posible dentro de los límites de token.</li>

<li><strong>FondoDINOFineTuning</strong>: Fine-tune <strong>Colocación de los puestos de control de DINO</strong> en conjuntos de datos personalizados.</li>
</ul><h2 id="example">Ejemplo de uso</h2>
A continuación se muestra un ejemplo de configuración YAML para <strong>Detección de objetos en cero</strong> utilizando <strong>Fondo DINO</strong>.


<details><summary><strong><span style="font-size: 1.4em;">Config</span></strong></summary>
</details>


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



<blockquote>

[!IMPORTANT]
La carpetaImageDatasetCV2, BBoxDrawer e Imagen Las plantillas Saver corresponden a las <a href="https://pypi.org/project/sinapsis-data-readers/">sinapsis-data-readers</a>, <a href="https://pypi.org/project/sinapsis-data-visualization/">sinapsis-data-visualization</a> y <a href="https://pypi.org/project/sinapsis-data-writers/">sinapsis-data-writers</a> paquetes, respectivamente. Si desea utilizar el ejemplo, asegúrese de instalar estos paquetes.

</blockquote>

Para ejecutar la configuración, utilice el CLI:

```bash
sinapsis run name_of_config.yml
```
<h2 id="webapp">🌐 Aplicación web</h2>
Laa <strong>Aplicaciones web de Sinapsis</strong> proporcionar una manera interactiva de explorar y experimentar con los modelos AI. Permiten a los usuarios generar salidas, probar diferentes entradas y visualizar resultados en tiempo real, facilitando la experiencia de las capacidades de cada modelo. A continuación se presentan las aplicaciones web disponibles e instrucciones para lanzarlas.

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
Si desea habilitar el intercambio de aplicaciones externas en Gradio, <code>export GRADIO_SHARE_APP=True</code>.

[!NOTE]
La configuración del agente puede cambiarse a través de la variable de ambiente  AGENT_CONFIG_PATH. Puede comprobar las configuraciones disponibles en cada carpeta de configuración de paquetes.

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
docker compose -f docker/compose_vision.yaml up sinapsis-huggingface-vision-gradio -d
```
<ol start="3">
<li><strong>Compruebe el estado</strong>:</li>
</ol>

```bash
docker logs -f sinapsis-huggingface-vision-gradio
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
docker compose -f docker/compose_vision.yaml down
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
uv run webapps/vision_demo.py
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

