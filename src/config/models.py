"""Config module."""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class AppConfig(BaseModel):
    """Pydantic model for application configuration."""

    output_dir: str
    config_path: str

class HTMLTagsConfig(BaseModel):
    """Pydantic model for HTML tag configuration."""

    sections: Dict[str, str]
    elements: Dict[str, Union[str, Dict[str, str]]]

class ModelsConfig(BaseModel):
    """Pydantic model for models such as docling models."""
    docling_path : str
    

class Config(BaseModel):
    """Pydantic model for the entire configuration."""

    app: AppConfig
    html_tags: HTMLTagsConfig
    models : ModelsConfig