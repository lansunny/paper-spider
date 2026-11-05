"""Read authoritative canonical pages; anchor identities do not depend on chunks."""
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pymupdf

from rag.formula_store import checked_directory, digest_file
from rag.ingestion import fingerprint, pdf_pages
from rag.storage import read_json
from rag.structural_chunking import ordered_blocks
from .schema import SOURCE_VERSION, SourceAnchor


def select_pilot(documents, limit, seed=42):
    """Title-based sampling buckets select the pilot, never graph communities."""
    anchors = ['2024.eacl-demo.16', '2024.naacl-long.20', '2023.emnlp-main.495',
               '2023.acl-long.557', '2020.emnlp-main.550', '2021.emnlp-main.224',
               '2025.findings-emnlp.568', '2025.findings-naacl.232']
    catalog = {p['paper_id']: p for p in documents}
    selected = [p for p in anchors if p in catalog][:limit]
    buckets = defaultdict(list)
    patterns = [('evaluation', r'evalua|benchmark|faithful|hallucin'),
                ('graph', r'graph|knowledge'), ('reasoning', r'reason|multi.hop|active|agent'),
                ('retrieval', r'dense|retriever|ranking|retrieval'),
                ('application', r'dialog|medical|legal|science|domain')]
    for paper in sorted(documents, key=lambda p: p['paper_id']):
        if paper['paper_id'] in selected:
            continue
        bucket = next((name for name, pattern in patterns if re.search(pattern, paper['title'], re.I)), 'other')
        buckets[bucket].append(paper['paper_id'])
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    while len(selected) < min(limit, len(catalog)):
        for name in sorted(buckets):
            if buckets[name] and len(selected) < limit:
                selected.append(buckets[name].pop())
    return [catalog[p] for p in selected]


def source_id(document_sha256, page, block_number, start, end, text_sha256):
    return 'S' + fingerprint([SOURCE_VERSION, document_sha256, page, block_number, start, end, text_sha256])[:20]


def load_sources(document, root):
    root = Path(root).resolve()
    pdf = (root / document['pdf_path']).resolve()
    if not pdf.is_relative_to(root) or digest_file(pdf) != document['sha256']:
        raise ValueError('PDF no longer matches the selected paper version')
    authority = read_json(root / 'runtime/formulas/CURRENT.json', {'papers': {}})['papers'].get(document['paper_id'])
    if authority:
        if authority['document_sha256'] != document['sha256']:
            raise ValueError('Active canonical text belongs to a different PDF version')
        directory = checked_directory(root / authority['directory'], root)
        if digest_file(directory / 'manifest.json') != authority['manifest_sha256']:
            raise ValueError('Authoritative formula manifest changed')
        manifest = read_json(directory / 'manifest.json')
        if (manifest['paper_id'] != document['paper_id'] or manifest['document_sha256'] != document['sha256']
                or manifest.get('status') != 'completed' or manifest['profile']['source_coordinates'] != SOURCE_VERSION
                or digest_file(directory / 'canonical_pages.jsonl') != manifest['canonical_pages_sha256']):
            raise ValueError('Invalid authoritative canonical pages')
        pages = [json.loads(s) for s in (directory / 'canonical_pages.jsonl').read_text(encoding='utf-8').splitlines() if s]
        origin = {'kind': 'formula_current', 'canonical_pages_sha256': manifest['canonical_pages_sha256']}
    else:
        raw = pdf_pages({**document, '_path': pdf}, root)
        pages = []
        for page in raw:
            text, records = ordered_blocks(page)
            pages.append({'pdf_page': page['pdf_page'], 'text': text,
                          'canonical_page_sha256': hashlib.sha256(text.encode()).hexdigest(),
                          'blocks': records})
        origin = {'kind': 'pdf_normalized_blocks', 'canonical_pages_sha256': fingerprint(pages)}
    sources = []
    with pymupdf.open(pdf) as source_pdf:
        if [p['pdf_page'] for p in pages] != list(range(1, len(source_pdf) + 1)):
            raise ValueError('Canonical pages must cover the entire PDF in order')
        for page in pages:
            text = page['text']
            page_hash = hashlib.sha256(text.encode()).hexdigest()
            if page_hash != page['canonical_page_sha256']:
                raise ValueError('Canonical page hash mismatch')
            # Reuse column/band reading order, preserving the original character coordinates.
            by_number = {b['number']: b for b in page['blocks']}
            raw_blocks = [{'block_number': b['number'], 'bbox': b['bbox'], 'text': b['text']}
                          for b in sorted(page['blocks'], key=lambda b: b['start'])]
            _, ordered = ordered_blocks({'blocks': raw_blocks, 'page_width': source_pdf[page['pdf_page'] - 1].rect.width})
            for record in ordered:
                block = by_number[record['number']]
                start, end = block['start'], block['end']
                if text[start:end] != block['text']:
                    raise ValueError('Canonical block interval mismatch')
                digest = hashlib.sha256(block['text'].encode()).hexdigest()
                sources.append(SourceAnchor(source_id=source_id(document['sha256'], page['pdf_page'],
                    block['number'], start, end, digest), paper_id=document['paper_id'],
                    document_sha256=document['sha256'], pdf_page=page['pdf_page'], block_number=block['number'],
                    source_start=start, source_end=end, canonical_page_sha256=page_hash,
                    text_sha256=digest, text=block['text'], bbox=block['bbox']).model_dump(mode='json'))
    if not sources or len({s['source_id'] for s in sources}) != len(sources):
        raise ValueError('Paper source text is empty or has duplicate anchors')
    return sources, origin


def source_batches(sources, max_chars):
    """Cover all source text; oversized blocks are exact slices, never silently omitted."""
    cards = []
    for source in sources:
        # A normal PDF block is far shorter; splitting retains exact canonical positions.
        for offset in range(0, len(source['text']), max_chars // 2):
            part = dict(source)
            part['text'] = source['text'][offset:offset + max_chars // 2]
            part['source_start'] += offset
            part['source_end'] = part['source_start'] + len(part['text'])
            part['text_sha256'] = hashlib.sha256(part['text'].encode()).hexdigest()
            part['source_id'] = source_id(part['document_sha256'], part['pdf_page'], part['block_number'],
                                          part['source_start'], part['source_end'], part['text_sha256'])
            cards.append(part)
    batches, current, used = [], [], 0
    for card in cards:
        size = len(json.dumps({'id': card['source_id'], 'page': card['pdf_page'], 'text': card['text']}, ensure_ascii=False))
        if current and used + size > max_chars:
            batches.append(current)
            current, used = [], 0
        current.append(card)
        used += size
    if current:
        batches.append(current)
    return cards, batches
