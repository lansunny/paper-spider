"""One retrievable chunk and its immutable provenance."""
import hashlib
import json
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceSpan(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    block_number: int
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    text_start: int = Field(ge=0)
    text_end: int = Field(gt=0)

    @model_validator(mode='after')
    def same_length(self):
        if self.source_end <= self.source_start or self.text_end <= self.text_start:
            raise ValueError('Source spans must have positive length')
        if self.source_end - self.source_start != self.text_end - self.text_start:
            raise ValueError('Copied source span lengths must match')
        return self


class Chunk(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    paper_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    chunker_version: str = Field(min_length=1)
    pdf_page: int | None = Field(default=None, ge=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    title: str = ''
    source_url: str = ''
    bboxes: tuple[tuple[float, float, float, float], ...] = ()
    paragraph_ids: tuple[str, ...] = ()
    corpus: str = 'papers'
    source_spans: tuple[SourceSpan, ...] = ()
    layout_column: str = ''
    layout_region: int = 0
    section_title: str = ''
    split_method: str = ''

    @field_validator('text')
    @classmethod
    def nonempty_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError('Chunk text must contain non-whitespace characters.')
        return text

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode('utf-8')).hexdigest()

    @property
    def chunk_id(self) -> str:
        identity = [self.corpus, self.paper_id, self.document_sha256, self.chunker_version,
                    self.pdf_page, self.paragraph_ids, self.chunk_index, self.text_sha256]
        # Keep legacy IDs byte-for-byte compatible with existing snapshots.
        if self.source_spans:
            identity.append([s.model_dump() for s in self.source_spans])
        return str(uuid5(NAMESPACE_URL, json.dumps(identity, ensure_ascii=False)))

    def payload(self) -> dict:
        return {**self.model_dump(mode='json'),
                'chunk_id': self.chunk_id, 'text_sha256': self.text_sha256}
