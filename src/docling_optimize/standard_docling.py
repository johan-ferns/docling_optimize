from pathlib import Path
import os
from docling_optimize.config.models import Config
from pydantic import BaseModel, ValidationError
import yaml
import logging

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions
)
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

from docling_optimize.pymupdf_backend import PyMuPDFDocumentBackend

logger = logging.getLogger("DoclingExtractor")


def _load_config(config_path: str | os.PathLike) -> Config:
    """Load configurations file."""
    path = Path(config_path)
    try:
        with path.open("r") as file:
            data = yaml.safe_load(file)
        return Config(**data)
    except (FileNotFoundError, ValidationError) as e:
        logger.error(f"Configuration error: {e}")
        raise


config_path = "src/docling_optimize/config/app_config.yml"


config = _load_config(config_path)

converter = DocumentConverter(
    format_options={

        InputFormat.PDF: PdfFormatOption(
            pipline_cls=StandardPdfPipeline, 
            pipeline_options=PdfPipelineOptions(artifacts_path=Path(config.models.docling_path),
                                                                    do_ocr=False, 
                                                                    force_backend_text=True,
                                                                    # accelerator_options=accelerator_options
                                                                    do_table_structure = True,
                                                                    generate_picture_images=True
                                                                    ),
            # backend = DoclingParseV4DocumentBackend
            backend = PyMuPDFDocumentBackend
        ),
    }
)


# filepath="../../data/pdf/inputs/example_1.pdf"

# doc = converter.convert(source=Path(filepath)).document

# print(len(doc.pictures))