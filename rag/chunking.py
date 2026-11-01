"""Token-bounded windows with overlap and retained page/block provenance."""
import re

from rag.schema import Chunk


def chunk_text(text, tokenizer, settings, *, spans=(), **metadata):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = encoded['offset_mapping']
    start, ordinal = 0, 0
    while start < len(offsets):
        end = min(start + settings.chunk_size, len(offsets))
        left, right = offsets[start][0], offsets[end - 1][1]
        content = text[left:right].strip()
        if content:
            boxes = tuple(box for begin, finish, box in spans if finish > left and begin < right)
            yield Chunk(**metadata, chunker_version=settings.chunker_version,
                        chunk_index=ordinal, text=content, bboxes=boxes)
            ordinal += 1
        if end == len(offsets):
            break
        start = end - settings.chunk_overlap


def normalized_blocks(blocks):
    # Preserve meaningful hyphens. Layout-aware ordering and table parsing are later experiments.
    pieces, spans, cursor = [], [], 0
    for block in blocks:
        text = re.sub(r'\s+', ' ', block['text']).strip()
        if not text:
            continue
        if pieces:
            cursor += 2
        spans.append((cursor, cursor + len(text), tuple(block['bbox'])))
        pieces.append(text)
        cursor += len(text)
    return '\n\n'.join(pieces), spans
