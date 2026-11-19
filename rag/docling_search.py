"""Search typed Docling views and resolve results back to the source object store."""
from collections import defaultdict
from contextlib import closing, nullcontext
from functools import cached_property
from hashlib import sha256
import json
import math
import re
import sqlite3
from threading import RLock
from typing import Literal

from rag.config import ROOT
from rag.docling_index import ObjectChunk, settings_for
from rag.docling_objects import mentions, target_for_number
from rag.embeddings import LocalModels
from rag.figure_vectors import FigureVectors, LocalClip, image_rgb, visual_directory
from rag.generation import select_context
from rag.ingestion import current_manifest, index_directory
from rag.lexical_v2 import LexicalIndexV2, lexical_tokens
from rag.query_decomposition import allocate_slots
from rag.query_planning_v2 import balanced_candidates
from rag.query_translation import HAN, translate_query
from rag.retrieval import reciprocal_rank_fusion, rerank_candidates
from rag.storage import read_json
from rag.vector_store import QdrantStore, models
from scripts.docling_prepare import digest

VIEWS = {'body': {'paragraph'}, 'caption': {'figure_text'}, 'tables': {'table_summary'}, 'rows': {'table_row'}}
BRANCH_DEPTH = 20
BODY_RERANK_DEPTH = 100
QueryLanguage = Literal['original', 'bm25_english', 'english']
ZH_OBJECT = re.compile(r'(图|表)\s*([A-Za-z]?\.?\d+(?:\.\d+)*[A-Za-z]?)(?![A-Za-z0-9])')


def fuse_objects(rankings, k):
    score, paths = defaultdict(float), defaultdict(dict)
    for branch, hits in rankings.items():
        rank = 0
        for hit in hits:
            oid = hit['object_id']
            if branch not in paths[oid]:
                rank += 1
                score[oid] += 1/(k+rank)
                paths[oid][branch] = {'rank': rank, **{key: hit[key] for key in (
                    'score', 'via', 'chunk_id', 'paragraph_id', 'matched_page') if key in hit}}
    return [{'object_id': oid, 'rrf_score': score[oid], 'paths': paths[oid]}
            for oid in sorted(score, key=lambda oid: (-score[oid], oid))]


def box_of(prov):
    box = prov.get('bbox') or {}
    if not {'l', 'r', 't', 'b', 'coord_origin'} <= box.keys():
        return None
    values = tuple(box[key] for key in ('l', 'r', 't', 'b'))
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return None
    l, r, t, b = values
    if box['coord_origin'] == 'BOTTOMLEFT':
        t, b = -t, -b
    elif box['coord_origin'] != 'TOPLEFT':
        return None
    return (l, t, r, b) if l < r and t < b else None


def nearest_figure(paragraph, figure, query_terms):
    if paragraph['paper_id'] != figure['paper_id'] or paragraph['pdf_sha256'] != figure['pdf_sha256']:
        return None
    caption_terms = set(lexical_tokens(figure.get('caption', '')+' '+figure.get('region_text', '')))
    semantic_hint = len(caption_terms & query_terms)/max(1, len(query_terms))
    same_section = bool(paragraph['headings'] and paragraph['headings'] == figure['headings'])
    best = None
    for p in paragraph['provenance']:
        for f in figure['provenance']:
            if p['page_no'] != f['page_no'] or (p.get('bbox') or {}).get('coord_origin') != (f.get('bbox') or {}).get('coord_origin'):
                continue
            a, b = box_of(p), box_of(f)
            if not a or not b:
                continue
            overlap = min(a[2], b[2])-max(a[0], b[0])
            if overlap < .25*min(a[2]-a[0], b[2]-b[0]):
                continue
            gap = max(b[1]-a[3], a[1]-b[3], 0)
            if gap > 180:
                continue
            candidate = (1-gap/200)*.60 + .20*same_section + .20*semantic_hint
            match = (candidate, p['page_no'])
            if best is None or match > best:
                best = match
    return best


def query_mentions(text):
    yield from mentions(text)
    for match in ZH_OBJECT.finditer(text):
        yield ('figure' if match[1] == '图' else 'table', match[2].upper(), match.start(), match.end())


def direct_object_query(text):
    ids = list(query_mentions(text))
    return len(ids) == 1 and ids[0][2] == 0 and not text[ids[0][3]:].strip(' .?？。')


class DoclingSearch:
    def __init__(self, profile, *, pilot=False, device='auto', body_rerank_k=BODY_RERANK_DEPTH):
        if not 1 <= body_rerank_k <= 200:
            raise ValueError('Body rerank depth must be between 1 and 200')
        _, self.cfg = settings_for(profile, pilot=pilot, device=device)
        self.body_rerank_k = body_rerank_k
        self.profile, self.pilot = profile, pilot
        self.manifest = current_manifest(self.cfg)
        if (not self.manifest or 'object_db' not in self.manifest
                or self.manifest['profile'] != self.cfg.index_profile()):
            raise RuntimeError('The requested Docling object index is not ready')
        self.db_path = (ROOT/self.manifest['object_db']).resolve()
        if not self.db_path.is_relative_to(ROOT/'runtime/docling-acl') or not self.db_path.is_file():
            raise RuntimeError('Docling object database is missing')
        self.text_model, self.clip = LocalModels(self.cfg), LocalClip('cpu')
        self.lock = RLock()
        self._visual, self._visual_key = None, None

    def connection(self):
        return sqlite3.connect(self.db_path.as_uri()+'?mode=ro', uri=True)

    @cached_property
    def lexical(self):
        path = index_directory(self.cfg)/(self.manifest['snapshot_id']+'.jsonl')
        if digest(path) != self.manifest['local_sha256']:
            raise ValueError('Docling local index checksum mismatch')
        sets = {branch: [] for branch in VIEWS}
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                chunk = ObjectChunk.model_validate_json(line)
                for branch, views in VIEWS.items():
                    if chunk.view in views:
                        sets[branch].append(chunk)
                        break
        return {branch: LexicalIndexV2(chunks) for branch, chunks in sets.items()}

    def visual(self):
        key = read_json(visual_directory(self.profile, self.pilot)/'CURRENT.json')
        with self.lock:
            if self._visual is None or self._visual_key != key:
                self._visual = FigureVectors(self.profile, self.manifest, self.pilot)
                self._visual_key = key
            return self._visual

    def _object(self, db, object_id):
        row = db.execute('SELECT artifact_dir,payload FROM objects WHERE object_id=?', (object_id,)).fetchone()
        if not row:
            raise KeyError(object_id)
        folder, payload = row
        obj = json.loads(payload)
        if self.manifest['documents'].get(obj['paper_id']) != obj['pdf_sha256']:
            raise KeyError(object_id)
        return folder, obj

    def load_object(self, object_id):
        with closing(self.connection()) as db:
            _, obj = self._object(db, object_id)
            if obj['kind'] in {'figure','table'}:
                obj['related_paragraphs'] = self._related_paragraphs(db, object_id)
            return self._public(obj, full=True)

    def _related_paragraphs(self, db, object_id):
        ids = db.execute('SELECT DISTINCT o.object_id, o.ordinal FROM links l JOIN objects o ON o.object_id=l.paragraph_id '
                         'WHERE l.object_id=? ORDER BY o.ordinal LIMIT 5', (object_id,))
        return [self._public(self._object(db, pid)[1]) for pid, _ in ids]

    @staticmethod
    def _public(obj, full=False):
        result = {key: obj[key] for key in ('object_id', 'paper_id', 'pdf_sha256', 'kind', 'headings', 'provenance', 'text')}
        for field in ('labels', 'caption', 'captions', 'footnotes', 'columns', 'related_paragraphs'):
            if field in obj:
                result[field] = obj[field]
        result['pages'] = sorted({p['page_no'] for p in obj['provenance']})
        result['crops'] = [{k: crop[k] for k in ('page', 'rect_pymupdf', 'width', 'height')} for crop in obj.get('crops', [])]
        if full and obj['kind'] == 'table':
            result.update(table=obj['table'], html=obj['html'], rows=obj['rows'])
        if full and obj['kind'] == 'figure':
            result['region_text'] = obj['region_text']
        return result

    def crop(self, object_id, index):
        with closing(self.connection()) as db:
            folder, obj = self._object(db, object_id)
        crops = obj.get('crops', [])
        if not 0 <= index < len(crops):
            raise KeyError(index)
        data = crops[index]
        base = (ROOT/folder).resolve()
        path = (base/data['path']).resolve()
        if not base.is_relative_to(ROOT/'runtime/docling-acl') or not path.is_relative_to(base):
            raise ValueError('Invalid crop location')
        blob = path.read_bytes()
        if sha256(blob).hexdigest() != data['sha256']:
            raise ValueError('Crop content no longer matches its object')
        return blob

    def _hybrid(self, branch, text, vector, store, scope, rerank_k, *, lexical_text=None, trace=None):
        ranked = self._recall(branch, text if lexical_text is None else lexical_text, vector, store,
                              scope, self.cfg.candidate_k, 'hybrid')
        if trace is not None:
            trace['union'] = ranked
            trace['rrf20'] = ranked[:20]
            trace['rerank_input'] = ranked[:rerank_k]
        ranked = rerank_candidates(self.text_model, text, ranked[:rerank_k], title=True)
        if trace is not None:
            trace['reranked'] = ranked
        return ranked

    def _recall(self, branch, text, vector, store, scope, limit, mode):
        paths = {}
        if mode in {'dense', 'hybrid'}:
            filters = store.snapshot_filter(self.manifest['snapshot_id'], scope, self.cfg.corpus)
            filters.must.append(models.FieldCondition(key='view', match=models.MatchAny(any=sorted(VIEWS[branch]))))
            points = store.client.query_points(
                self.cfg.collection, query=vector.tolist(), using=self.cfg.vector_name,
                query_filter=filters, limit=limit, with_payload=True, with_vectors=False).points
            paths['dense'] = [{**p.payload, 'score': float(p.score), 'scores': {'dense': float(p.score)}} for p in points]
        if mode in {'bm25', 'hybrid'}:
            paths['bm25'] = self.lexical[branch].search(text, limit, scope)
        return reciprocal_rank_fusion(paths, self.cfg.rrf_k) if mode == 'hybrid' else paths[mode]

    def retrieve_body(self, question, *, paper_ids=None, facets=(), mode='hybrid', rerank=True):
        """Return ranked paragraph chunks for the caller's evidence budget."""
        if mode not in {'bm25', 'dense', 'hybrid'} or not question.strip():
            raise ValueError('需要有效的正文检索方式和问题。')
        if paper_ids and not set(paper_ids).issubset(self.manifest['documents']):
            raise ValueError('选择的论文尚未进入 Docling 正文索引。')
        if len(facets) >= min(self.cfg.candidate_k, self.body_rerank_k):
            facets = ()
        queries = [(question, paper_ids), *[(facet.question, [facet.paper_id]) for facet in facets]]
        texts = [self._query_texts(query.strip(), 'english') for query, _ in queries]
        quotas = allocate_slots(self.cfg.candidate_k, len(queries))
        store_context = closing(QdrantStore(self.cfg)) if mode != 'bm25' else nullcontext()
        with self.lock, store_context as store:
            if mode != 'bm25':
                store.validate_collection()
            pools = []
            for (rank_text, lexical_text, _), (_, scope), limit in zip(texts, queries, quotas, strict=True):
                vector = self.text_model.query_vector(rank_text) if mode != 'bm25' else None
                pools.append(self._recall('body', lexical_text, vector, store, scope, limit, mode))
            if facets:
                fused = reciprocal_rank_fusion({f'query{i}': hits for i, hits in enumerate(pools)}, self.cfg.rrf_k)
                candidates = balanced_candidates(pools[0], pools[1:], fused, self.body_rerank_k)
            else:
                candidates = pools[0][:self.body_rerank_k]
            ranked = rerank_candidates(self.text_model, texts[0][0], candidates, title=True) if rerank else candidates
        if facets and rerank:
            front = ranked[:1]
            for facet in facets:
                best = next((hit for hit in ranked if hit['paper_id'] == facet.paper_id), None)
                if best is not None and best not in front:
                    front.append(best)
            promoted = {hit['chunk_id'] for hit in front}
            ranked = [*front, *(hit for hit in ranked if hit['chunk_id'] not in promoted)]
        return ranked, {'rerank_pairs': len(candidates) if rerank else 0,
                        'translations': [meta for _, _, meta in texts if meta]}

    @staticmethod
    def _query_texts(text, language):
        if language == 'original' or not HAN.search(text):
            return text, text, None
        try:
            translation = translate_query(text)
        except RuntimeError:
            return text, text, {'status': 'fallback', 'mode': language}
        english = translation.english
        return (english if language == 'english' else text), english, {
            'status': 'translated', 'mode': language, 'model': translation.model}

    def _body_results(self, db, hits):
        for hit in hits:
            source = self._public(self._object(db, hit['object_id'])[1])
            yield {**source, 'text': hit['text'], 'chunk_id': hit['chunk_id'], 'view_span': hit['view_span'],
                   'score': hit['score'], 'linked_object_ids': hit['linked_object_ids'],
                   'title': hit['title'], 'matched_page': hit['pdf_page'],
                   **({'citation_id': hit['citation_id']} if 'citation_id' in hit else {})}

    def _labels(self, db, text, paper):
        if not paper:
            return []
        out = {}
        for kind, number, *_ in query_mentions(text):
            matches = target_for_number(number, lambda n: db.execute(
                'SELECT object_id FROM labels WHERE paper_id=? AND kind=? AND label=?', (paper,kind,n)).fetchall())
            if len(matches) == 1:
                out[matches[0][0]] = kind
        return list(out.items())

    def _context_figures(self, db, text, body_hits):
        terms = set(self.lexical['caption'].query(text)['tokens'])
        explicit, nearby, cache, seen = [], [], {}, set()
        for rank, hit in enumerate(body_hits[:8], 1):
            pid = hit['object_id']
            if hit['chunk_id'] in seen:
                continue
            seen.add(hit['chunk_id'])
            paragraph = self._object(db, pid)[1]
            left, right = hit['view_span']
            if any(kind == 'figure' and left < end and right > start for kind, _, start, end in mentions(paragraph['text'])):
                for oid, in db.execute("SELECT DISTINCT o.object_id FROM links l JOIN objects o ON o.object_id=l.object_id "
                                       "WHERE l.paragraph_id=? AND o.kind='figure' AND l.start<? AND l.end>? ORDER BY o.object_id",
                                       (pid, right, left)):
                    explicit.append({'object_id': oid, 'via': 'explicit_reference', 'chunk_id': hit['chunk_id'], 'paragraph_id': pid})
                continue
            if hit['paper_id'] not in cache:
                cache[hit['paper_id']] = [json.loads(payload) for payload, in db.execute(
                    "SELECT payload FROM objects WHERE paper_id=? AND kind='figure'", (hit['paper_id'],))]
            relevant = {**paragraph, 'provenance': [p for p in paragraph['provenance']
                        if not p.get('charspan') or left < p['charspan'][1] and right > p['charspan'][0]]}
            candidates = []
            for fig in cache[hit['paper_id']]:
                if not fig.get('crops'):
                    continue
                match = nearest_figure(relevant, fig, terms)
                if match is not None:
                    score, page = match
                    candidates.append({'object_id': fig['object_id'], 'via': 'nearby_candidate',
                                       'chunk_id': hit['chunk_id'], 'paragraph_id': pid, 'matched_page': page,
                                       'score': score/(1+.10*(rank-1))})
            nearby.extend(sorted(candidates, key=lambda item: (-item['score'], item['object_id']))[:2])
        return explicit + sorted(nearby, key=lambda item: (-item['score'], item['object_id']))

    def search(self, text='', *, paper_id=None, paper_ids=None, kinds=('body','figure','table'), top_k=5,
               clip_text=None, query_image=None, visual=True, nearby=True, trace=False,
               query_language: QueryLanguage = 'english'):
        scope = list(dict.fromkeys(paper_ids or ([paper_id] if paper_id else [])))
        if unknown := set(scope) - self.manifest.get('documents', {}).keys():
            raise KeyError(next(iter(unknown)))
        exact_paper = scope[0] if len(scope) == 1 else None
        body, captions, summaries, rows = [], [], [], []
        effective = text.strip()
        response = {'profile': self.profile, 'snapshot_id': self.manifest['snapshot_id'], 'body': [],
                    'figures': [], 'tables': [], 'visual': 'disabled'}
        if 'body' in kinds:
            response.update(body_context=[], body_context_chars=0, body_context_budget=self.cfg.context_char_budget)
        if trace:
            response['rankings'] = {}
        # Resolve the optional network step before taking the local inference lock.
        search_text, lexical_text, translation = self._query_texts(
            effective, 'original' if direct_object_query(effective) else query_language)
        if translation:
            response['query_translation'] = translation
        with self.lock, closing(self.connection()) as db:
            exact = self._labels(db, effective, exact_paper)
            direct_lookup = (any(kind in kinds for _, kind in exact) and direct_object_query(effective)
                             and not clip_text and query_image is None)
            if effective and not direct_lookup:
                vector = self.text_model.query_vector(search_text)
                with closing(QdrantStore(self.cfg)) as store:
                    store.validate_collection()
                    for branch in ('body','caption','tables','rows'):
                        if (branch == 'body' and ({'body', 'figure'} & set(kinds)) or
                                branch == 'caption' and 'figure' in kinds or
                                branch in {'tables','rows'} and 'table' in kinds):
                            body_trace = response['rankings'].setdefault('body', {}) if trace and branch == 'body' else None
                            depth = self.body_rerank_k if branch == 'body' and 'body' in kinds else self.cfg.rerank_k
                            ranked = self._hybrid(branch, search_text, vector, store, scope or None, depth,
                                                  lexical_text=lexical_text, trace=body_trace)
                            {'body': body, 'caption': captions, 'tables': summaries, 'rows': rows}[branch].extend(
                                ranked if branch == 'body' else ranked[:BRANCH_DEPTH])
            if 'body' in kinds:
                context, used = select_context(body, self.cfg.context_char_budget)
                response['body'] = list(self._body_results(db, body[:top_k]))
                response['body_context'] = list(self._body_results(db, context))
                response['body_context_chars'] = used
            if 'figure' in kinds:
                context = self._context_figures(db, lexical_text, body) if body else []
                rankings = {'caption': captions, 'context': [h for h in context if nearby or h['via'] == 'explicit_reference']}
                if visual and not direct_lookup and (effective or clip_text or query_image is not None):
                    image = image_rgb(query_image) if query_image is not None else None
                    try:
                        index = self.visual()
                        query = self.clip.images([image])[0] if image is not None else self.clip.text(clip_text or search_text)
                    except RuntimeError:
                        if query_image is not None:
                            raise
                        response['visual'] = 'unavailable'
                    else:
                        rankings['clip'] = index.rank(query, BRANCH_DEPTH, paper_ids=scope)
                        response['visual'] = 'image' if image is not None else 'text'
                if trace:
                    response['rankings']['figure'] = {branch: [
                        {k: hit[k] for k in ('object_id', 'score', 'via') if k in hit} for hit in hits]
                        for branch, hits in rankings.items()}
                figures = self._with_exact(db, exact, 'figure', rankings, top_k)
                for item in figures:
                    item['related_paragraphs'] = self._related_paragraphs(db, item['object_id'])
                    via = item['paths'].get('context', {})
                    if via.get('via') == 'nearby_candidate':
                        item['nearby_paragraph'] = self._public(self._object(db, via['paragraph_id'])[1])
                        item['nearby_paragraph']['matched_page'] = via['matched_page']
                response['figures'] = figures
            if 'table' in kinds:
                if trace:
                    response['rankings']['table'] = {branch: [{k: h[k] for k in ('object_id', 'row') if k in h} for h in hits]
                                                     for branch, hits in (('table',summaries),('row',rows))}
                tables = self._with_exact(db, exact, 'table', {'table': summaries, 'row': rows}, top_k)
                for item in tables:
                    raw = self._object(db, item['object_id'])[1]
                    item['related_paragraphs'] = self._related_paragraphs(db, item['object_id'])
                    by_row = {r['row']: r for r in raw['rows']}
                    indexes = list(dict.fromkeys(h['row'] for h in rows if h['object_id'] == item['object_id']))[:3]
                    item['matched_rows'] = [{**by_row[i], 'cells': [raw['table']['table_cells'][j]
                         for j in by_row[i]['cell_ids']]} for i in indexes]
                    item['header_cells'] = [cell for cell in raw['table']['table_cells'] if cell.get('column_header')]
                    item['html'] = raw['html']
                response['tables'] = tables
        return response

    def _with_exact(self, db, exact, kind, rankings, limit):
        fused = fuse_objects(rankings, self.cfg.rrf_k)
        by_id = {h['object_id']: h for h in fused}
        explicit = [oid for oid, label in exact if label == kind]
        ordered = list(dict.fromkeys([*explicit, *by_id]))
        return [{**self._public(self._object(db, oid)[1]), **by_id.get(oid, {'paths': {}}), 'exact_number': oid in explicit}
                for oid in ordered[:limit]]


def profiles():
    results = []
    for path in sorted((ROOT/'runtime/indexes').glob('paperrag_docling_acl_*/CURRENT.json')):
        match = re.fullmatch(r'paperrag_docling_acl_([a-f0-9]{16})(_pilot)?', path.parent.name)
        if not match:
            continue
        marker = read_json(path)
        if not marker or not marker.get('object_db'):
            continue
        profile, pilot = match[1], bool(match[2])
        visual = read_json(visual_directory(profile, pilot)/'CURRENT.json') or {}
        results.append({'profile': profile, 'pilot': pilot, 'papers': len(marker['documents']), 'snapshot_id': marker['snapshot_id'],
                        'clip_ready': visual.get('index_snapshot') == marker['snapshot_id']})
    return results
