"""Formula candidates and OCR provenance, independent of frozen text chunks."""
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Box = tuple[float, float, float, float]


class FormulaSourceText(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    pdf_page: int = Field(ge=1)
    block_number: int
    bbox: Box
    text: str
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    canonical_page_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    relation: Literal['preceding', 'following', 'overlapping', 'explicit_reference', 'formula_body']

    @model_validator(mode='after')
    def valid_source(self):
        if self.source_end-self.source_start != len(self.text):
            raise ValueError('Context must remain an exact canonical-source substring')
        return self


class FormulaElement(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    element_id: str
    element_type: Literal['formula'] = 'formula'
    paper_id: str
    document_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    title: str
    source_url: str = ''
    pdf_page: int = Field(ge=1)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    page_rotation: int
    coordinate_system: Literal['unrotated_page_points'] = 'unrotated_page_points'
    bbox: Box
    crop_bbox: Box
    crop_scope: Literal['formula_region', 'mixed_paragraph_context'] = 'formula_region'
    equation_number: str | None = None
    layout_kind: Literal['numbered_display', 'unnumbered_display', 'inline_or_mixed', 'numbered_raster']
    detection_method: str
    confidence_band: Literal['high', 'medium', 'low']
    confidence_note: str = 'Heuristic ranking band, not a calibrated probability.'
    detection_evidence: list[str] = Field(default_factory=list)
    source_spans: list[FormulaSourceText] = Field(default_factory=list)
    canonical_page_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_line_ids: list[str]
    raw_text: str
    raw_text_note: str = 'PDF reading order can lose fraction, superscript and symbol structure; this is NOT LaTeX.'
    context: list[FormulaSourceText]
    references: list[FormulaSourceText] = Field(default_factory=list)
    reference_resolution: str = 'unresolved'
    section_title: str | None = None
    section_method: str = 'heading heuristic; may be incomplete'
    search_text: str
    search_text_origin: Literal['source_text_proxy'] = 'source_text_proxy'
    proxy_trust: Literal['unverified_pdf_text'] = 'unverified_pdf_text'
    proxy_for_retrieval_only: Literal[True] = True
    image_path: str
    image_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    render_dpi: int = Field(ge=72, le=400)
    review_status: Literal['unreviewed'] = 'unreviewed'
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def valid_geometry(self):
        for box in [self.bbox, self.crop_bbox]:
            if (not all(math.isfinite(v) for v in box) or not 0 <= box[0] < box[2] <= self.page_width+.01
                    or not 0 <= box[1] < box[3] <= self.page_height+.01):
                raise ValueError('Formula box must lie inside the unrotated page')
        if not (self.crop_bbox[0] <= self.bbox[0] and self.crop_bbox[1] <= self.bbox[1]
                and self.crop_bbox[2] >= self.bbox[2] and self.crop_bbox[3] >= self.bbox[3]):
            raise ValueError('Rendered crop must contain the formula box')
        return self


class FormulaOCRResult(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    element_id: str
    document_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    image_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    engine: str = Field(min_length=1, max_length=100)
    engine_revision: str = Field(min_length=1, max_length=200)
    latex: str = Field(min_length=1, max_length=12000)
    status: Literal['recognized_unverified'] = 'recognized_unverified'

    @model_validator(mode='after')
    def real_transcription(self):
        if not self.latex.strip() or '```' in self.latex:
            raise ValueError('OCR must return a nonempty LaTeX string without code fences')
        return self


class FormulaAnnotation(BaseModel):
    """Gold labels refer to immutable PDF/source coordinates, never extracted IDs."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    annotation_id: str
    paper_id: str
    document_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    pdf_page: int = Field(ge=1)
    bbox: Box
    label: Literal['formula', 'not_formula', 'not_equation_number']
    equation_number: str | None = None
    canonical_page_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    source_quote: str
    group: str
    review_basis: str

    @model_validator(mode='after')
    def stable_source_anchor(self):
        if self.source_end-self.source_start != len(self.source_quote):
            raise ValueError('Annotation source quote must match its interval')
        if not all(math.isfinite(v) for v in self.bbox) or self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError('Annotation must have a valid original-page rectangle')
        return self
