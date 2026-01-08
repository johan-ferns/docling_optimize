#Extractor class for docling to convert PDF to markdown

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions
)
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from pathlib import Path
import math
import yaml
import os
from pydantic import ValidationError
import logging


from extraction_models import (
    Coordinates,
    Extracted,
    ExtractionResults,
)
from sphinx.rules.types import ExtractedType
from pymupdf_backend import PyMuPDFDocumentBackend


class DoclingExtractor(Extractor):

    def __init__(
            self,
            config_path: str = "src/docling_optimize/config/app_config.yaml",
    ):
        
        # model_path = "//FLD6FILER/DScD/Special Project/Sphinx/docling/models"
        self.logger = logging.getLogger("DoclingExtractor")
        self.config = self._load_config(config_path)
        self.converter = DocumentConverter(
            format_options={

                InputFormat.PDF: PdfFormatOption(
                    pipline_cls=StandardPdfPipeline, 
                    pipeline_options=PdfPipelineOptions(artifacts_path=Path(self.config.models.docling_path),
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

    def get_converter(self):
        """Provide generated converter."""
        return self.converter

    def _load_config(self, config_path: str | os.PathLike) -> Config:
        """Load configurations file."""
        path = Path(config_path)
        try:
            with path.open("r") as file:
                data = yaml.safe_load(file)
            return Config(**data)
        except (FileNotFoundError, ValidationError) as e:
            self.logger.error(f"Configuration error: {e}")
            raise

    def get_page_no(self, item):
        """Get page number of item."""
        prov = item.prov[0]
        return prov.page_no

    def get_bbox_coordinates(self, doc, item):
        """Convert Box coordinates for use."""
        prov = item.prov[0]
        page = doc.pages[prov.page_no]
        bbox = prov.bbox.to_top_left_origin(page.size.height)
        bbox_coordinate = Coordinates(x1=math.ceil(bbox.l), 
                                    y1=math.ceil(bbox.t), 
                                    x2=math.ceil(bbox.r), 
                                    y2=math.ceil(bbox.b))
        
        return bbox_coordinate

    def extract(self, path: str) -> ExtractionResults:
        """
        Extract data from the PDF file.

        Args:
        ----
            path (str): Path to the file or URL to process.

        Returns:
        -------
            ExtractionResults: Extracted data and metadata.
        """
        
        doc = self.converter.convert(source=Path(path)).document
        serializer = MarkdownDocSerializer(doc=doc)

        extracted_texts, extracted_tables, extracted_figures = [], [], []

        for idx, text_item in enumerate(doc.texts):    
            extracted_texts.append(Extracted(
                        extracted_type=ExtractedType.TEXT,
                        page_no=self.get_page_no(item=text_item),
                        path=None,
                        coordinates=self.get_bbox_coordinates(doc=doc, item=text_item),
                        data=serializer.serialize(item=text_item).text,
                        data_type="text"
            ))


        # For figures
        for idx, pic_item in enumerate(doc.pictures):    
            extracted_figures.append(Extracted(
                        extracted_type=ExtractedType.FIGURE,
                        page_no=self.get_page_no(item=pic_item),
                        path=None,
                        coordinates=self.get_bbox_coordinates(doc=doc, item=pic_item),
                        data=pic_item.image.uri._url.__str__(),
                        data_type=pic_item.image.mimetype
            ))


        # For tables
        for idx, table_item in enumerate(doc.tables):    
            extracted_tables.append(Extracted(
                        extracted_type=ExtractedType.TABLE,
                        page_no=self.get_page_no(item=table_item),
                        path=None,
                        coordinates=self.get_bbox_coordinates(doc=doc, item=table_item),
                        data=serializer.serialize(item=table_item).text,
                        data_type="text"
            ))

        return ExtractionResults(
            texts=extracted_texts,
            tables=extracted_tables,
            figures=extracted_figures
        )        