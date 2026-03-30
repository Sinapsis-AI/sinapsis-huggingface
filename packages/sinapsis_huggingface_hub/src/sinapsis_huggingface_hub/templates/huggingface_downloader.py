# -*- coding: utf-8 -*-

from typing import Literal

from huggingface_hub import snapshot_download
from pydantic import Field
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import TemplateAttributes, TemplateAttributeType, UIPropertiesMetadata
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR

from sinapsis_huggingface_hub.helpers.tags import Tags


class HuggingFaceDownloaderAttributes(TemplateAttributes):
    """Defines the configuration for downloading a repository snapshot from the Hugging Face Hub.

    Attributes:
        repo_id (str): The repository ID from the Hub (e.g., "stabilityai/stable-diffusion-2-1").
        repo_type (Literal["dataset", "space", "model"]): The type of the repository.
        revision (str | None): The specific model version to use (e.g., a branch name, tag, or commit hash).
        cache_dir (str): The directory where downloaded files will be cached.
        allow_patterns (list[str] | str | None): A pattern or list of patterns to specify which files to download.
        ignore_patterns (list[str] | str | None): A pattern or list of patterns to specify which files to ignore.
        max_workers (int): The maximum number of threads to use for parallel downloads.
        force_download (bool): If True, forces the repository to be re-downloaded even if it's already cached.
        proxies (dict): A dictionary of proxy servers to use for the download.
        local_files_only (bool): If True, the function will only look for the files in the cache and will not
            access the network.
        etag_timeout (float): The timeout in seconds for checking the ETag of the files.
        resume_download (bool): If True, resumes an interrupted download.
    """

    repo_id: str
    repo_type: Literal["dataset", "space", "model"] = "model"
    revision: str | None = None
    cache_dir: str = str(SINAPSIS_CACHE_DIR)
    allow_patterns: list[str] | str | None = None
    ignore_patterns: list[str] | str | None = None
    max_workers: int = 8
    force_download: bool = False
    proxies: dict = Field(default_factory=dict)
    local_files_only: bool = False
    etag_timeout: float = 10.0
    resume_download: bool = False


class HuggingFaceDownloader(Template):
    """A Sinapsis Template that downloads a repository snapshot from the Hugging Face Hub."""

    AttributesBaseModel = HuggingFaceDownloaderAttributes
    UIProperties = UIPropertiesMetadata(
        category="Hugging Face Hub",
        tags=[Tags.HUGGINGFACE, Tags.MODELS],
    )

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.logger.info(f"Starting download of {self.attributes.repo_id} to {self.attributes.cache_dir}")
        self.path = snapshot_download(**self.attributes.model_dump(exclude_none=True, exclude={"metadata"}))

    def execute(self, container: DataContainer) -> DataContainer:
        """Injects the local directory path of the downloaded repository into the data container.

        Args:
            container (DataContainer): The data container to which the directory path will be added.

        Returns:
            DataContainer: The updated data container, now containing the directory path.
        """
        self._set_generic_data(container, self.path)
        return container
