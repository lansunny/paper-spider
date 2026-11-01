"""PDF/QASPER ingestion, reusable embeddings, and atomic index snapshots."""
import hashlib
import json
import os
import tempfile
import math
import warnings
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import portalocker
import pymupdf

from rag.chunking import chunk_text, normalized_blocks
from rag.config import ROOT
from rag.schema import Chunk
from rag.storage import atomic_json, atomic_text, read_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def index_directory(settings, root=ROOT):
    return root / 'runtime/indexes' / settings.collection


def current_manifest(settings, root=ROOT):
    return read_json(index_directory(settings, root) / 'CURRENT.json')


def current_chunks(settings, root=ROOT):
    manifest = current_manifest(settings, root)
    if not manifest:
        raise RuntimeError('尚未建立索引，请先点击建立索引，或运行 main.py --task index。')
    if manifest['profile'] != settings.index_profile():
        raise ValueError('当前索引与模型/分块配置不一致，请为新配置使用独立 collection 并建立索引。')
    path = index_directory(settings, root) / (manifest['snapshot_id'] + '.jsonl')
    chunks = [Chunk.model_validate_json(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    if len(chunks) != manifest['chunks'] or fingerprint([c.chunk_id for c in chunks]) != manifest['chunk_ids_hash']:
        raise ValueError('Local index snapshot integrity check failed; rebuild the index.')
    return manifest, chunks


def paper_catalog(root=ROOT):
    originals = read_json(root / 'data/paper_manifest.json', [])
    uploads = read_json(root / 'runtime/uploads/manifest.json', [])
    return [p for p in originals if p.get('status') == 'downloaded'] + uploads


def source_documents(settings, root=ROOT):
    if settings.corpus == 'papers':
        documents = []
        for paper in paper_catalog(root):
            path = (root / paper['pdf_path']).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f"PDF missing or outside workspace: {paper['paper_id']}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            documents.append({**paper, 'sha256': digest, '_path': path})
        return documents
    split = settings.corpus.removeprefix('qasper-')
    path = root / 'data/starter' / f'{split}_corpus.jsonl'
    documents = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    return [{**paper, 'sha256': fingerprint(paper)} for paper in documents]


def pdf_pages(document, root=ROOT):
    cache = root / 'runtime/parsed' / (document['sha256'] + '.jsonl')
    cache.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(cache) + '.lock', timeout=60), pymupdf.open(document['_path']) as pdf:
        if pdf.needs_pass:
            raise ValueError('Encrypted PDF cannot be indexed.')
        if cache.exists():
            try:
                rows = [json.loads(line) for line in cache.read_text(encoding='utf-8').splitlines()]
                if len(rows) != len(pdf) or [p['pdf_page'] for p in rows] != list(range(1, len(pdf) + 1)):
                    raise ValueError('Incomplete or unordered cached pages')
                for row in rows:
                    if not isinstance(row['blocks'], list):
                        raise ValueError('Invalid cached blocks')
                    numbers = set()
                    for b in row['blocks']:
                        if (not isinstance(b['text'], str) or not isinstance(b['block_number'], int)
                                or b['block_number'] in numbers or len(b['bbox']) != 4
                                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in b['bbox'])):
                            raise ValueError('Invalid cached block')
                        numbers.add(b['block_number'])
                # Older caches lack dimensions. Enrich metadata, never reorder text.
                if any('page_width' not in p or 'page_height' not in p for p in rows):
                    for row, page in zip(rows, pdf, strict=True):
                        row.update(page_width=page.rect.width, page_height=page.rect.height)
                    atomic_text(cache, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
                return rows
            except (ValueError, KeyError, TypeError, UnicodeError) as error:
                quarantine = cache.parent / 'quarantine' / (cache.stem + '-' + uuid4().hex + '.jsonl')
                quarantine.parent.mkdir(exist_ok=True)
                cache.replace(quarantine)
                warnings.warn(f"Reparsing damaged cache for {document['paper_id']}: {type(error).__name__}")
        rows = []
        for index, page in enumerate(pdf):
            blocks = [{'bbox': list(b[:4]), 'text': b[4], 'block_number': b[5]}
                      for b in page.get_text('blocks', sort=False) if b[6] == 0]
            rows.append({'pdf_page': index + 1, 'page_width': page.rect.width,
                         'page_height': page.rect.height, 'blocks': blocks})
        if not any(b['text'].strip() for page in rows for b in page['blocks']):
            raise ValueError(f"{document['paper_id']} 没有可提取的文字层，需要 OCR 后再入库。")
        atomic_text(cache, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        return rows


def document_chunks(document, tokenizer, settings, root=ROOT):
    common = {'paper_id': document['paper_id'], 'document_sha256': document['sha256'],
              'title': document['title'], 'corpus': settings.corpus}
    if settings.corpus == 'papers':
        common['source_url'] = document.get('landing_url') or document.get('source_url') or ''
        for page in pdf_pages(document, root):
            if settings.chunk_strategy == 'layout-blocks':
                from rag.structural_chunking import structural_chunks
                yield from structural_chunks(page, tokenizer, settings, **common)
                continue
            text, spans = normalized_blocks(page['blocks'])
            yield from chunk_text(text, tokenizer, settings, spans=spans,
                                  pdf_page=page['pdf_page'], **common)
    else:
        # Labels/answers are deliberately not read by the indexer.
        common['source_url'] = document.get('source', '')
        for paragraph in document['paragraphs']:
            yield from chunk_text(paragraph['text'], tokenizer, settings,
                                  paragraph_ids=(paragraph['paragraph_id'],), **common)


def cached_vectors(chunks, models, settings, root=ROOT):
    profile = {k: settings.index_profile()[k] for k in
               ['embedding_model', 'embedding_revision', 'vector_size', 'embedding_max_length', 'pooling', 'normalized']}
    if settings.chunk_strategy == 'layout-blocks':
        profile['embedding_input'] = 'body-v1'
    directory = root / 'runtime/embedding_cache' / fingerprint(profile)
    directory.mkdir(parents=True, exist_ok=True)
    values, missing, hits = [None] * len(chunks), [], 0
    for index, chunk in enumerate(chunks):
        # Cache the actual model input. Body-only v1 intentionally equals text.
        target = directory / (hashlib.sha256(embedding_input(chunk).encode()).hexdigest() + '.npy')
        try:
            vector = np.load(target, allow_pickle=False)
            if vector.shape != (settings.vector_size,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=0.01):
                raise ValueError('Invalid cached vector')
            values[index] = vector
            hits += 1
        except (OSError, ValueError):
            missing.append(index)
    if missing:
        encoded = models.encode([embedding_input(chunks[index]) for index in missing])
        for index, vector in zip(missing, encoded, strict=True):
            values[index] = vector
            target = directory / (hashlib.sha256(embedding_input(chunks[index]).encode()).hexdigest() + '.npy')
            descriptor, temporary = tempfile.mkstemp(dir=directory, suffix='.npy')
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    np.save(stream, vector, allow_pickle=False)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    return np.asarray(values, dtype=np.float32), hits


def embedding_input(chunk):
    """Only body-v1 is supported; future prefixes require a new profile version."""
    return chunk.text


def build_index(settings, models, store, root=ROOT, progress=None, force=False):
    progress = progress or (lambda **state: None)
    directory = index_directory(settings, root)
    directory.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(directory / 'build.lock'), timeout=0):
        documents = source_documents(settings, root)
        if not documents:
            raise ValueError('No documents available for indexing.')
        document_hashes = {d['paper_id']: d['sha256'] for d in documents}
        old = current_manifest(settings, root)
        store.ensure_collection()
        if (not force and old and old['profile'] == settings.index_profile()
                and old['documents'] == document_hashes and store.count(old['snapshot_id']) == old['chunks']):
            current_chunks(settings, root)  # also validate the local side before reusing it
            return {**old, 'reused': True}
        chunks = []
        for index, document in enumerate(documents, 1):
            progress(stage='chunking', completed=index - 1, total=len(documents))
            chunks.extend(document_chunks(document, models.tokenizer, settings, root))
        if not chunks:
            raise ValueError('No text chunks produced.')
        ids_hash = fingerprint([chunk.chunk_id for chunk in chunks])
        snapshot = fingerprint({'profile': settings.index_profile(), 'ids': ids_hash})[:32]
        cache_hits = 0
        for start in range(0, len(chunks), 64):
            batch = chunks[start:start + 64]
            vectors, hits = cached_vectors(batch, models, settings, root)
            cache_hits += hits
            store.upsert(batch, vectors, snapshot)
            progress(stage='embedding_and_upsert', completed=min(start + 64, len(chunks)), total=len(chunks))
        if store.count(snapshot) != len(chunks):
            raise RuntimeError('Qdrant count mismatch; old snapshot remains active.')
        atomic_text(directory / (snapshot + '.jsonl'), ''.join(chunk.model_dump_json() + '\n' for chunk in chunks))
        manifest = {'snapshot_id': snapshot, 'profile': settings.index_profile(), 'documents': document_hashes,
                    'chunks': len(chunks), 'chunk_ids_hash': ids_hash, 'embedding_cache_hits': cache_hits,
                    'built_at': datetime.now(timezone.utc).isoformat()}
        atomic_json(directory / 'CURRENT.json', manifest)
        progress(stage='ready', completed=len(chunks), total=len(chunks))
        return {**manifest, 'reused': False}
