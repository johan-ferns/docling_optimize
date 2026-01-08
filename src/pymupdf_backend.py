import logging
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import pymupdf  # PyMuPDF
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from PIL import Image

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.backend_options import PdfBackendOptions

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


def get_pdf_page_geometry(
    page: pymupdf.Page,
    angle: float = 0.0,
    boundary_type: PdfPageBoundaryType = PdfPageBoundaryType.CROP_BOX,
) -> PdfPageGeometry:
    """
    Create PdfPageGeometry from a PyMuPDF Page object.

    Args:
        page: PyMuPDF Page object
        angle: Page rotation angle in degrees (default: 0.0)
        boundary_type: The boundary type for the page (default: CROP_BOX)

    Returns:
        PdfPageGeometry with all the different bounding boxes properly set
    """
    # Get the page rectangle (similar to crop_box in pypdfium2)
    rect = page.rect
    bbox = BoundingBox(
        l=rect.x0,
        b=rect.y0,
        r=rect.x1,
        t=rect.y1,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )

    # Get different page boxes from PyMuPDF
    # PyMuPDF Page objects have standard PDF boxes: mediabox and cropbox
    try:
        mediabox = page.mediabox
        cropbox = page.cropbox
    except AttributeError:
        # Fallback if page boxes are not available (shouldn't happen with PyMuPDF >= 1.23.0)
        mediabox = rect
        cropbox = rect

    # For boxes not directly available, use the main rect as fallback
    media_bbox = BoundingBox(
        l=mediabox.x0,
        b=mediabox.y0,
        r=mediabox.x1,
        t=mediabox.y1,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )

    crop_bbox = BoundingBox(
        l=cropbox.x0,
        b=cropbox.y0,
        r=cropbox.x1,
        t=cropbox.y1,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )

    # Use bbox as fallback for boxes not available in PyMuPDF
    art_bbox = bbox
    bleed_bbox = bbox
    trim_bbox = bbox

    return PdfPageGeometry(
        angle=angle,
        rect=BoundingRectangle.from_bounding_box(bbox),
        boundary_type=boundary_type,
        art_bbox=art_bbox,
        bleed_bbox=bleed_bbox,
        crop_bbox=crop_bbox,
        media_bbox=media_bbox,
        trim_bbox=trim_bbox,
    )


class PyMuPDFPageBackend(PdfPageBackend):
    def __init__(self, doc: pymupdf.Document, document_hash: str, page_no: int):
        self.valid = True
        try:
            self._page: pymupdf.Page = doc.load_page(page_no)
        except Exception:
            _log.info(
                f"An exception occurred when loading page {page_no} of document {document_hash}.",
                exc_info=True,
            )
            self.valid = False
        self._text_dict: Optional[dict] = None

    def is_valid(self) -> bool:
        return self.valid

    def _compute_text_cells(self) -> List[TextCell]:
        """Compute text cells from PyMuPDF."""
        if not self._text_dict:
            # Get text with detailed positioning information
            self._text_dict = self._page.get_text("dict")

        cells = []
        cell_counter = 0
        page_size = self.get_size()

        # Extract text blocks from the dictionary
        for block in self._text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    # Get the bounding box for the line
                    line_bbox = line.get("bbox")
                    if not line_bbox:
                        continue

                    x0, y0, x1, y1 = line_bbox

                    # Extract text from spans
                    text_pieces = []
                    for span in line.get("spans", []):
                        text_pieces.append(span.get("text", ""))

                    line_text = "".join(text_pieces)
                    if not line_text.strip():
                        continue

                    # PyMuPDF uses top-left origin, convert to bottom-left for consistency
                    cells.append(
                        TextCell(
                            index=cell_counter,
                            text=line_text,
                            orig=line_text,
                            from_ocr=False,
                            rect=BoundingRectangle.from_bounding_box(
                                BoundingBox(
                                    l=x0,
                                    b=page_size.height
                                    - y1,  # Convert from top-left to bottom-left
                                    r=x1,
                                    t=page_size.height
                                    - y0,  # Convert from top-left to bottom-left
                                    coord_origin=CoordOrigin.BOTTOMLEFT,
                                )
                            ).to_top_left_origin(page_size.height),
                        )
                    )
                    cell_counter += 1

        return cells

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        """Get bounding boxes of images on the page."""
        AREA_THRESHOLD = 0  # Can be adjusted similar to pypdfium2
        page_size = self.get_size()

        # Get image list from the page
        image_list = self._page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                # Get image bounding box
                # img[0] is the xref of the image
                xref = img[0]
                # Get all instances of this image on the page
                img_rects = self._page.get_image_rects(xref)

                for rect in img_rects:
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1

                    # Convert from top-left to bottom-left origin
                    cropbox = BoundingBox(
                        l=x0,
                        t=page_size.height - y0,
                        r=x1,
                        b=page_size.height - y1,
                        coord_origin=CoordOrigin.TOPLEFT,
                    )

                    if cropbox.area() > AREA_THRESHOLD:
                        cropbox = cropbox.scaled(scale=scale)
                        yield cropbox
            except Exception as e:
                _log.debug(f"Error getting image rect: {e}")
                continue

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        """Extract text within a bounding box."""
        page_size = self.get_size()

        # Convert to top-left origin if needed (PyMuPDF uses top-left)
        if bbox.coord_origin != CoordOrigin.TOPLEFT:
            bbox = bbox.to_top_left_origin(page_size.height)

        # Create PyMuPDF rect
        rect = pymupdf.Rect(bbox.l, bbox.t, bbox.r, bbox.b)

        # Extract text from the rectangle
        text = self._page.get_text("text", clip=rect)
        return text

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        """Get segmented page with text cells and geometry."""
        if not self.valid:
            return None

        text_cells = self._compute_text_cells()

        # Get the PDF page geometry
        dimension = get_pdf_page_geometry(self._page)

        # Create SegmentedPdfPage
        return SegmentedPdfPage(
            dimension=dimension,
            textline_cells=text_cells,
            char_cells=[],
            word_cells=[],
            has_textlines=len(text_cells) > 0,
            has_words=False,
            has_chars=False,
        )

    def get_text_cells(self) -> Iterable[TextCell]:
        """Get all text cells from the page."""
        return self._compute_text_cells()

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        """Render the page as an image."""
        page_size = self.get_size()

        if not cropbox:
            cropbox = BoundingBox(
                l=0,
                r=page_size.width,
                t=0,
                b=page_size.height,
                coord_origin=CoordOrigin.TOPLEFT,
            )

        # Convert cropbox to top-left origin if needed
        if cropbox.coord_origin != CoordOrigin.TOPLEFT:
            cropbox = cropbox.to_top_left_origin(page_size.height)

        # Create a matrix for scaling and cropping
        mat = pymupdf.Matrix(scale, scale)

        # Define the clip rectangle
        clip = pymupdf.Rect(cropbox.l, cropbox.t, cropbox.r, cropbox.b)

        # Render the page
        pix = self._page.get_pixmap(matrix=mat, clip=clip)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        return img

    def get_size(self) -> Size:
        """Get the page size."""
        rect = self._page.rect
        return Size(width=rect.width, height=rect.height)

    def unload(self):
        """Unload the page resources."""
        if self.valid and self._page is not None:
            self._page = None
        self._text_dict = None


class PyMuPDFDocumentBackend(PdfDocumentBackend):
    def __init__(
        self,
        in_doc: "InputDocument",
        path_or_stream: Union[BytesIO, Path],
        options: PdfBackendOptions = PdfBackendOptions(),
    ):
        super().__init__(in_doc, path_or_stream, options)

        password = (
            self.options.password.get_secret_value() if self.options.password else None
        )

        try:
            _log.info("Parsing the document with PyMuPDF")
            if isinstance(path_or_stream, BytesIO):
                # Read from BytesIO
                self._doc = pymupdf.open(stream=path_or_stream.read(), filetype="pdf")
            else:
                # Read from file path
                self._doc = pymupdf.open(path_or_stream)

            # Apply password if provided
            if password and self._doc.needs_pass:
                if not self._doc.authenticate(password):
                    raise RuntimeError(
                        f"Invalid password for document with hash {self.document_hash}"
                    )
        except Exception as e:
            raise RuntimeError(
                f"PyMuPDF could not load document with hash {self.document_hash}"
            ) from e

    def page_count(self) -> int:
        """Get the number of pages in the document."""
        return len(self._doc)

    def load_page(self, page_no: int) -> PyMuPDFPageBackend:
        """Load a specific page."""
        return PyMuPDFPageBackend(self._doc, self.document_hash, page_no)

    def is_valid(self) -> bool:
        """Check if the document is valid."""
        return self.page_count() > 0

    def unload(self):
        """Unload the document and free resources."""
        super().unload()
        if self._doc:
            self._doc.close()
            self._doc = None