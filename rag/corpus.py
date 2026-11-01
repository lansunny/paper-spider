"""Adapt the current PDF baseline to versioned chunks without changing its text."""
import json
from collections.abc import Iterator
from pathlib import Path

from rag.config import ROOT
from rag.schema import Chunk

CHUNKER_VERSION = 'blocks-1800-v1'


def iter_chunks(root: Path = ROOT, paper_ids: set[str] | None = None) -> Iterator[Chunk]:
    catalog = {p['paper_id']: p for p in json.loads(
        (root / 'data/paper_manifest.json').read_text(encoding='utf-8'))}
    for source in sorted((root / 'data/parsed').glob('*.jsonl')):
        with source.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    page = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'{source.name}:{line_number}: expected one PDF page per JSONL line') from error
                if paper_ids is not None and page['paper_id'] not in paper_ids:
                    continue
                title = catalog.get(page['paper_id'], {}).get('title', '')
                buffer, boxes, ordinal = '', [], 0

                def make_chunk() -> Chunk:
                    return Chunk(paper_id=page['paper_id'], document_sha256=page['pdf_sha256'],
                                 chunker_version=CHUNKER_VERSION, pdf_page=page['pdf_page'],
                                 chunk_index=ordinal, text=buffer, title=title,
                                 source_url=page['source_url'], bboxes=tuple(boxes))

                for block in page['blocks']:
                    if buffer and len(buffer) + len(block['text']) > 1800:
                        if buffer.strip():
                            yield make_chunk()
                            ordinal += 1
                        buffer, boxes = '', []
                    buffer += block['text'] + '\n'
                    boxes.append(tuple(block['bbox']))
                if buffer.strip():
                    yield make_chunk()
