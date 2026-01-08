import os
from typing import Literal
from enum import Enum
from pydantic import BaseModel, Field



class ExtractedType(str, Enum):
    TABLE = "table"
    FIGURE = "figure"
    TEXT = "text"


class Language(str, Enum):
    FRENCH = "french"
    ENGLISH = "english"



class Coordinates(BaseModel):
    x1: int = Field(default=0, ge=0)
    y1: int = Field(default=0, ge=0)
    x2: int = Field(default=0, ge=0)
    y2: int = Field(default=0, ge=0)


class Extracted(BaseModel):
    extracted_type: ExtractedType = Field(ExtractedType.TEXT)
    page_no: int | None = None
    path: os.PathLike | None = None
    coordinates: Coordinates | None = None
    data: bytes | str | None = None
    data_type: Literal["image/png", "image/jpg", "application/pdf", "text"] = Field("text")


class ExtractionResults(BaseModel):
    texts: list[Extracted]
    tables: list[Extracted]
    figures: list[Extracted]
