"""Validated paper descriptions and immutable, canonical-text source anchors."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DIMENSIONS = ('topic', 'method', 'experiment', 'conclusion')
LABELS = dict(zip(DIMENSIONS, ('主题', '方法', '实验', '结论')))
FIELDS = {
    'topic': ('research_objects', 'problems', 'tasks', 'scenarios'),
    'method': ('core_idea', 'modules', 'training', 'inference'),
    'experiment': ('datasets', 'baselines', 'metrics', 'settings', 'results'),
    'conclusion': ('findings', 'conditions', 'limitations'),
}
SOURCE_VERSION = 'normalized_blocks-v1'


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class SourceAnchor(StrictModel):
    source_id: str
    paper_id: str
    document_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_version: Literal['normalized_blocks-v1'] = SOURCE_VERSION
    pdf_page: int = Field(ge=1)
    block_number: int
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    canonical_page_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    text_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    text: str = Field(min_length=1)
    bbox: tuple[float, float, float, float]

    @model_validator(mode='after')
    def exact_interval(self):
        import hashlib
        import math
        if self.source_end - self.source_start != len(self.text):
            raise ValueError('Source text must exactly fill its canonical interval')
        if hashlib.sha256(self.text.encode()).hexdigest() != self.text_sha256:
            raise ValueError('Source text hash mismatch')
        if (not all(math.isfinite(v) for v in self.bbox)
                or self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]):
            raise ValueError('Invalid original PDF rectangle')
        return self


class Claim(StrictModel):
    field: str
    text: str = Field(min_length=1, max_length=600)
    source_ids: list[str] = Field(min_length=1, max_length=32)


class DimensionDescription(StrictModel):
    status: Literal['available', 'not_found_in_input']
    description: str = Field(max_length=1600)
    claims: list[Claim] = Field(max_length=16)
    reason: str = Field(default='', max_length=350)

    @model_validator(mode='after')
    def consistent_status(self):
        if self.status == 'available' and (not self.description.strip() or not self.claims):
            raise ValueError('Available dimensions need a description and sourced claims')
        if self.status == 'not_found_in_input' and (self.description or self.claims or not self.reason.strip()):
            raise ValueError('Missing dimensions need a reason, empty description and empty claims')
        return self


class PaperDescription(StrictModel):
    topic: DimensionDescription
    method: DimensionDescription
    experiment: DimensionDescription
    conclusion: DimensionDescription

    @model_validator(mode='after')
    def known_fields(self):
        for dimension in DIMENSIONS:
            for claim in getattr(self, dimension).claims:
                if claim.field not in FIELDS[dimension]:
                    raise ValueError(f'Unknown {dimension} field: {claim.field}')
        return self


def validate_description(value, allowed_sources):
    result = PaperDescription.model_validate(value)
    allowed_sources = set(allowed_sources)
    for dimension in DIMENSIONS:
        for claim in getattr(result, dimension).claims:
            if len(set(claim.source_ids)) != len(claim.source_ids):
                raise ValueError(f'{dimension}.{claim.field}: duplicate source IDs; cite each source once')
            unknown = set(claim.source_ids) - allowed_sources
            if unknown:
                raise ValueError(f'{dimension}.{claim.field}: unknown source IDs {sorted(unknown)}; '
                                 'copy an exact ID from this input or omit the unsupported claim')
    return result
