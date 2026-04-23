<h1 align="center"><br/><br/><a href="https://sinapsis.tech/"><img alt="" src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true" width="300"/></a><br/>Sinapsis Hugging Face Hub
<br/></h1>

Sinapsis Hugging Face Hub proporciona un sistema sencillo y flexible <strong>no código</strong> aplicación de la <strong>Hugging Face Hub</strong> biblioteca. Permite a los usuarios gestionar fácilmente modelos, conjuntos de datos y espacios para tareas relacionadas con Hugging Face.

<p align="center"><a href="#installation">🐍 Instalación</a> •
<a href="#features">📦 Características</a> •
<a href="#example">↑ Uso del ejemplo</a> •
<a href="#documentation">📙 Documentación</a> •
<a href="#license">🔍 Licencia</a>

<h2 id="installation">🐍 Instalación</h2>

Instala el administrador de paquetes de tu elección. Alentamos el uso de <code>uv</code>


Ejemplo con <code>uv</code>:



```bash
  uv pip install sinapsis-huggingface-hub --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
  pip install sinapsis-huggingface-hub --extra-index-url https://pypi.sinapsis.tech
```

<blockquote>

[!IMPORTANT]
Las plantillas pueden requerir dependencias opcionales adicionales. Para el desarrollo, recomendamos instalar el paquete con todas las dependencias opcionales:

con <code>uv</code>:

</blockquote>

```bash
  uv pip install sinapsis-huggingface-hub[all] --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
  pip install sinapsis-huggingface-hub[all] --extra-index-url https://pypi.sinapsis.tech
```
<h2 id="features">📦 Características</h2>
Las plantillas en este paquete incluyen funcionalidad para:
<ul>
<li><strong>HuggingFaceDownloader</strong>: Descarga una instantánea del repositorio del Hugging Face Hub.</li>
</ul><h2 id="example">Ejemplo de uso</h2>
A continuación se muestra un ejemplo de configuración YAML para ejecutar un <strong>Stable Diffusion Downloader</strong> tubería con Sinapsis.


<details><summary><strong><span style="font-size: 1.4em;">Config</span></strong></summary>
</details>


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

