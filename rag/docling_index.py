"""Independent dense/lexical ACL index for versioned Docling artifacts."""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import re
import sqlite3
from uuid import NAMESPACE_URL, uuid5

import portalocker
from pydantic import Field

from rag.config import ROOT, RetrievalSettings
from rag.docling_objects import mentions, target_for_number
from rag.embeddings import LocalModels
from rag.ingestion import current_manifest, fingerprint, index_directory
from rag.lexical_v2 import LexicalIndexV2
from rag.retrieval import reciprocal_rank_fusion, rerank_candidates
from rag.schema import Chunk
from rag.storage import atomic_json, atomic_text, read_json
from rag.vector_store import QdrantStore
from scripts.docling_prepare import BASE, BGE, WORK, artifact, digest, sources, stable_hash


class DoclingSettings(RetrievalSettings):
    preparation_id: str
    docling_format: int = 2

    @property
    def chunker_version(self):
        return ('docling-paragraph-' if self.docling_format >= 3 else 'docling-hybrid-') + self.preparation_id

    def index_profile(self):
        return {**super().index_profile(), 'provenance_version': 'docling-objects-v1' if self.docling_format >= 3 else 'docling-elements-v1',
                'embedding_input': 'docling-object-view-v1' if self.docling_format >= 3 else 'docling-contextualize-v1'}


class DoclingChunk(Chunk):
    embedding_text: str
    tokens: int
    headings: list[str]
    captions: list[str] = Field(default_factory=list)
    kind: str
    elements: list[dict]
    pages: list[int]


class ObjectChunk(DoclingChunk):
    object_id: str
    view: str
    row: int | None = None
    view_span: tuple[int, int]
    linked_object_ids: list[str] = Field(default_factory=list)
    linked_paragraph_ids: list[str] = Field(default_factory=list)


def settings_for(profile_id: str, *, pilot=False, device='auto'):
    if not re.fullmatch(r'[a-f0-9]{16}', profile_id):
        raise ValueError('Invalid Docling preparation profile ID')
    profile = read_json(WORK / profile_id / 'profile.json')
    if not profile or stable_hash(profile)[:16] != profile_id:
        raise ValueError('Docling profile does not match its ID')
    if profile['bge'] != read_json(BGE / 'paperrag-model.json') or profile['tokenizer_sha256'] != digest(BGE / 'tokenizer.json'):
        raise ValueError('Local BGE artifacts differ from those used for Docling chunking')
    base = read_json(ROOT / 'configs/retrieval_v2.json')
    suffix = '_pilot' if pilot else ''
    settings = DoclingSettings(**{**base, 'collection': f'paperrag_docling_acl_{profile_id}{suffix}',
                                  'qdrant_url': os.getenv('QDRANT_URL') or base['qdrant_url'],
                                  'chunk_size': profile['options_chunker']['max_tokens'], 'chunk_overlap': 0,
                                  'device': device, 'preparation_id': profile_id, 'docling_format': profile['format_version']})
    return profile, settings


def object_snapshot(profile_id, ids, baseline, pilot, *, from_full=False):
    directory = WORK / profile_id
    entry = read_json(directory / ('OBJECTS_PILOT.json' if pilot and not from_full else 'OBJECTS_CURRENT.json'))
    if not entry or entry['profile_id'] != profile_id or entry['baseline_snapshot'] != baseline['snapshot_id']:
        raise ValueError('Object corpus was not published for the current baseline')
    path = directory / (entry['snapshot_id'] + '.sqlite3')
    if not path.is_file():
        raise ValueError('Object database is missing')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as db:
        db_papers = dict(db.execute('SELECT paper_id,pdf_sha256 FROM papers'))
    expected = baseline['documents'] if from_full else {pid: baseline['documents'][pid] for pid in ids}
    if db_papers != expected:
        raise ValueError('Object snapshot differs from the requested papers')
    return path.relative_to(ROOT).as_posix(), entry['snapshot_id']


def require_object_smoke(profile_id, baseline, snapshot_id):
    directory = WORK/profile_id
    validated = read_json(directory/'object-validation.json') or {}
    tested = read_json(directory/'object-smoke.json') or {}
    expected = (profile_id, baseline['snapshot_id'], snapshot_id)

    def scope(report):
        return tuple(report.get(key) for key in ('profile', 'baseline_snapshot', 'object_snapshot_id'))

    if (scope(validated) != expected or validated.get('failures') != []
            or validated.get('checked_papers') != len(baseline['documents'])):
        raise ValueError('Validate the complete Docling object snapshot before building vectors')
    if scope(tested) != expected or tested.get('status') != 'passed' or not tested.get('checks'):
        raise ValueError('Run the small-batch smoke test on this complete Docling snapshot before building vectors')


def prepared_chunks(ids, baseline, catalog, profile_id, profile, settings):
    for pid in ids:
        sha = baseline['documents'][pid]
        pdf = (ROOT / catalog[pid]['pdf_path']).resolve()
        if not pdf.is_relative_to(ROOT.resolve()) or digest(pdf) != sha:
            raise ValueError(f'PDF no longer matches the Docling artifacts: {pid}')
        folder = artifact(profile_id, pid, sha)
        info = read_json(folder / 'manifest.json')
        path = folder / 'chunks.jsonl'
        if not info or (info['paper_id'], info['profile_id'], info['pdf_sha256']) != (pid, profile_id, sha):
            raise ValueError(f'No completed Docling preparation for {pid}')
        parsed = artifact('parsed/' + profile['parse_id'], pid, sha) / 'document.json'
        if digest(path) != info['sha256']['chunks.jsonl'] or digest(parsed) != info['sha256']['document.json']:
            raise ValueError(f'Docling artifact checksum mismatch: {pid}')
        # JSONL records end at LF, not at Unicode page/paragraph separators in PDF text.
        with path.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not rows or len(rows) != info['chunks']:
            raise ValueError(f'Incomplete Docling chunks: {pid}')
        for index, row in enumerate(rows):
            if row['chunk_index'] != index or not 0 < row['tokens'] <= settings.embedding_max_length or not row['pages']:
                raise ValueError(f'Invalid Docling chunk: {pid}/{index}')
            chunk_type = ObjectChunk if settings.docling_format >= 3 else DoclingChunk
            yield chunk_type(paper_id=pid, document_sha256=sha, chunker_version=settings.chunker_version,
                               pdf_page=row['pages'][0], title=catalog[pid]['title'],
                               source_url=catalog[pid].get('landing_url', ''), section_title=' / '.join(row['headings']),
                               paragraph_ids=tuple(e['id'] for e in row['elements']), split_method=profile['options_chunker']['algorithm'], **row)


def build(profile_id, *, paper_ids=(), device='auto', from_full_objects=False):
    profile, cfg = settings_for(profile_id, pilot=bool(paper_ids), device=device)
    baseline, catalog = sources()
    ids = sorted(paper_ids or catalog)
    if len(ids) != len(set(ids)) or not set(ids).issubset(catalog):
        raise ValueError('Expected unique paper IDs from the active ACL baseline')
    if from_full_objects and (not paper_ids or cfg.docling_format < 3):
        raise ValueError('A pilot subset is required when testing complete Docling objects')
    directory = index_directory(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(directory / 'build.lock'), timeout=0):
        object_db, object_snapshot_id = (object_snapshot(profile_id, ids, baseline, bool(paper_ids), from_full=from_full_objects)
                                         if cfg.docling_format >= 3 else (None, None))
        if object_db and not paper_ids:
            require_object_smoke(profile_id, baseline, object_snapshot_id)
        chunks = list(prepared_chunks(ids, baseline, catalog, profile_id, profile, cfg))
        ids_hash = fingerprint([c.chunk_id for c in chunks])
        input_hash = fingerprint([c.embedding_text for c in chunks])
        fields = [cfg.index_profile(), ids_hash, input_hash]
        snapshot = fingerprint([*fields, object_snapshot_id] if object_db else fields)[:32]
        shard = directory / (snapshot + '.jsonl')
        shard_text = ''.join(c.model_dump_json() + '\n' for c in chunks)
        local_hash = sha256(shard_text.encode('utf-8')).hexdigest()
        documents = {pid: baseline['documents'][pid] for pid in ids}
        print(f'{cfg.collection}: {len(documents)} PDFs, {len(chunks)} chunks', flush=True)
        with closing(QdrantStore(cfg)) as store:
            store.ensure_collection()
            old = current_manifest(cfg)
            if (old and old['snapshot_id'] == snapshot and old['documents'] == documents and shard.is_file()
                    and digest(shard) == local_hash and store.count(snapshot) == len(chunks)):
                return old
            model = LocalModels(cfg)
            for start in range(0, len(chunks), 64):
                batch = chunks[start:start + 64]
                # Intermediate points are invisible until the new collection's manifest is published.
                by_id = {str(uuid5(NAMESPACE_URL, snapshot + ':' + c.chunk_id)): c for c in batch}
                existing = store.client.retrieve(cfg.collection, list(by_id), with_payload=False, with_vectors=False)
                present = {str(point.id) for point in existing}
                missing = [c for point, c in by_id.items() if point not in present]
                if missing:
                    store.upsert(missing, model.encode([chunk.embedding_text for chunk in missing]), snapshot)
                if start % 512 == 0 or start + 64 >= len(chunks):
                    print(f'Vectors {min(start + 64, len(chunks))}/{len(chunks)}', flush=True)
            if store.count(snapshot) != len(chunks):
                raise RuntimeError('Docling snapshot point count mismatch')
        latest = read_json(BASE)
        if latest['snapshot_id'] != baseline['snapshot_id'] or latest['documents'] != baseline['documents']:
            raise RuntimeError('ACL baseline changed during build; isolated snapshot was not published')
        atomic_text(shard, shard_text)
        manifest = {'snapshot_id': snapshot, 'collection': cfg.collection, 'profile': cfg.index_profile(),
                    'baseline_snapshot': baseline['snapshot_id'], 'documents': documents, 'chunks': len(chunks),
                    'chunk_ids_hash': ids_hash, 'local_sha256': local_hash, 'embedding_inputs_sha256': input_hash,
                    'chunk_kinds': dict(Counter(c.kind for c in chunks)), 'built_at': datetime.now(timezone.utc).isoformat()}
        if object_db:
            manifest.update(object_db=object_db, object_snapshot_id=object_snapshot_id)
        atomic_json(directory / 'CURRENT.json', manifest)
        return manifest


def linked_evidence(db, hits):
    """Retrieve via stable IDs; do not inject the entire linked graph."""
    result = {}
    for hit in hits:
        object_id = hit['object_id']
        row = db.execute('SELECT payload,artifact_dir FROM objects WHERE object_id=?', (object_id,)).fetchone()
        if row is None:
            raise ValueError('Chunk points to an object missing from the database')
        obj = json.loads(row[0])
        if obj['paper_id'] != hit['paper_id'] or obj['pdf_sha256'] != hit['document_sha256']:
            raise ValueError('Index hit and object PDF provenance disagree')
        if object_id not in result:
            result[object_id] = {'kind': obj['kind'], 'paper_id': obj['paper_id'], 'pdf_sha256': obj['pdf_sha256'],
                                 'source_pages': sorted({p['page_no'] for p in obj['provenance']}),
                                 'labels': obj.get('labels', []), 'caption': obj.get('caption', ''),
                                 'crops': [{**c, 'path': f'{row[1]}/{c["path"]}'} for c in obj.get('crops', [])]}
        linked = (hit.get('linked_object_ids', []) if obj['kind'] == 'paragraph' else hit.get('linked_paragraph_ids', []))
        hit['related'] = []
        for linked_id in linked[:5]:
            record = db.execute('SELECT payload,artifact_dir FROM objects WHERE object_id=?', (linked_id,)).fetchone()
            if record is None:
                raise ValueError('Linked object is missing from the database')
            target, folder = json.loads(record[0]), record[1]
            if target['paper_id'] != obj['paper_id'] or target['pdf_sha256'] != obj['pdf_sha256']:
                raise ValueError('Linked object escaped its source PDF')
            hit['related'].append({'object_id': linked_id, 'kind': target['kind'], 'paper_id': target['paper_id'],
                 'pages': sorted({p['page_no'] for p in target['provenance']}), 'labels': target.get('labels', []),
                 'text': target.get('caption') or target['text'],
                 'crops': [{**c, 'path': f'{folder}/{c["path"]}'} for c in target.get('crops', [])]})
        if obj['kind'] == 'table':
            table = result[object_id]
            table.update(columns=obj['columns'], footnotes=obj['footnotes'])
            if hit.get('row') is not None:
                row_view = next(r for r in obj['rows'] if r['row'] == hit['row'])
                hits_cells = [obj['table']['table_cells'][i] for i in row_view['cell_ids']]
                hit['table_row'] = {**row_view, 'cells': hits_cells}
    return result


def search(profile_id, question, *, pilot=False, device='auto', top_k=5, rerank=True, paper_id=None, models=None):
    _, cfg = settings_for(profile_id, pilot=pilot, device=device)
    manifest = current_manifest(cfg)
    if not manifest or manifest['profile'] != cfg.index_profile():
        raise ValueError('The requested isolated Docling index has not been built')
    if paper_id and paper_id not in manifest['documents']:
        raise ValueError('Paper does not exist in this Docling snapshot')
    path = index_directory(cfg) / (manifest['snapshot_id'] + '.jsonl')
    if digest(path) != manifest['local_sha256']:
        raise ValueError('Docling local index checksum mismatch')
    chunk_type = ObjectChunk if cfg.docling_format >= 3 else DoclingChunk
    with path.open(encoding='utf-8') as stream:
        chunks = [chunk_type.model_validate_json(line) for line in stream if line.strip()]
    scope = [paper_id] if paper_id else None
    bm25 = LexicalIndexV2(chunks).search(question, cfg.candidate_k, scope)
    model = models if models is not None else LocalModels(cfg)
    with closing(QdrantStore(cfg)) as store:
        store.validate_collection()
        dense = store.search(model.query_vector(question), manifest['snapshot_id'], cfg.candidate_k, scope)
    ranked = reciprocal_rank_fusion({'bm25': bm25, 'dense': dense}, cfg.rrf_k)
    if rerank:
        ranked = rerank_candidates(model, question, ranked[:cfg.rerank_k], title=True)
    response = {'collection': cfg.collection, 'snapshot_id': manifest['snapshot_id']}
    if 'object_db' in manifest:
        path = ROOT / manifest['object_db']
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as db:
            if paper_id:
                # Number lookups only have meaning inside a known paper. A duplicate label is not a match.
                exact = []
                for kind, number, *_ in mentions(question):
                    objects = target_for_number(number, lambda label: db.execute(
                        'SELECT object_id FROM labels WHERE paper_id=? AND kind=? AND label=?',
                        (paper_id, kind, label)).fetchall())
                    if len(objects) == 1:
                        match = next((c for c in chunks if getattr(c, 'object_id', None) == objects[0][0]
                                      and c.view in {'table_summary', 'figure_text'}), None)
                        if match:
                            exact.append({**match.payload(), 'retrieval_origin': 'object_number'})
                exact = {h['chunk_id']: h for h in exact}
                ranked = [*exact.values(), *(h for h in ranked if h['chunk_id'] not in exact)]
            hits = ranked[:top_k]
            response.update(object_db=manifest['object_db'], objects=linked_evidence(db, hits), hits=hits)
    else:
        response['hits'] = ranked[:top_k]
    return response
