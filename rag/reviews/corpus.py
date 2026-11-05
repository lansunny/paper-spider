"""Pinned graph claims, four community memberships and exact original-page reads."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from rag.formula_store import digest_file
from rag.ingestion import fingerprint, paper_catalog
from rag.paper_graph.schema import DIMENSIONS
from rag.paper_graph.sources import load_sources
from rag.paper_graph.store import PaperGraphStore


class ReviewCorpus:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.store = PaperGraphStore(self.root)
        bundle = self.store.bundle()
        if not bundle:
            raise ValueError('请先建立论文四维图。')
        self.directory, self.manifest = bundle
        self.papers, self.claims, self.sources = {}, {}, {}
        self.memberships = {d: {} for d in DIMENSIONS}
        self.communities = {d: [] for d in DIMENSIONS}
        self.profiles = {}
        with closing(self.store._connection(bundle)) as connection:
            self.papers = {r['paper_id']: dict(r) for r in connection.execute('SELECT * FROM papers')}
            for row in connection.execute('SELECT * FROM profiles ORDER BY paper_id,dimension'):
                profile = dict(row)
                profile['claims'] = []
                for claim in json.loads(row['claims_json']):
                    identifier = 'C' + fingerprint([row['paper_id'], self.papers[row['paper_id']]['document_sha256'],
                        row['dimension'], claim['field'], claim['text'], sorted(claim['source_ids'])])[:24]
                    entry = {**claim, 'claim_id': identifier, 'paper_id': row['paper_id'], 'dimension': row['dimension']}
                    self.claims[identifier] = entry
                    profile['claims'].append(entry)
                self.profiles[(row['paper_id'], row['dimension'])] = profile
            wanted = {sid for c in self.claims.values() for sid in c['source_ids']}
            for row in connection.execute('SELECT * FROM sources'):
                if row['source_id'] in wanted:
                    self.sources[row['source_id']] = json.loads(row['anchor_json'])
            for row in connection.execute('SELECT * FROM nodes'):
                self.memberships[row['dimension']][row['paper_id']] = row['community_id']
            for row in connection.execute('SELECT * FROM communities'):
                self.communities[row['dimension']].append(json.loads(row['data_json']))
        if wanted - self.sources.keys():
            raise ValueError('A graph claim points to a missing original-source anchor')
        for dimension in DIMENSIONS:
            self.communities[dimension].sort(key=lambda c: c['ordinal'])
        if digest_file(self.directory / 'vectors.npz') != self.manifest['artifacts']['vectors.npz']:
            raise ValueError('Paper dimension vectors changed')
        with np.load(self.directory / 'vectors.npz', allow_pickle=False) as arrays:
            self.vectors = {d: (arrays[d+'_paper_ids'].tolist(), arrays[d+'_vectors'].copy()) for d in DIMENSIONS}
        self.documents = {p['paper_id']: p for p in paper_catalog(self.root) if p['paper_id'] in self.papers}
        self._pages = {}

    @property
    def build_id(self):
        return self.manifest['build_id']

    def assert_current(self):
        if self.store.summary().get('build_id') != self.build_id:
            raise ValueError('论文图已更新，请更新研究卡片后重试。')

    def paper_claims(self, paper_id, dimension=None):
        return [c for d in ([dimension] if dimension else DIMENSIONS)
                for c in self.profiles[(paper_id, d)]['claims']]

    def canonical_pages(self, paper_id):
        document = next((p for p in paper_catalog(self.root) if p['paper_id'] == paper_id), None)
        if not document or document['sha256'] != self.papers[paper_id]['document_sha256']:
            raise ValueError('论文版本已变化，请更新图和研究卡片。')
        path = (self.root/document['pdf_path']).resolve()
        if not path.is_relative_to(self.root) or digest_file(path) != document['sha256']:
            raise ValueError('原始 PDF 已变化，不能沿旧坐标取证。')
        if paper_id not in self._pages:
            blocks, _ = load_sources(document, self.root)
            grouped = {}
            for block in blocks:
                grouped.setdefault(block['pdf_page'], []).append(block)
            pages = {}
            for page, records in grouped.items():
                records.sort(key=lambda b: b['source_start'])
                text = '\n\n'.join(b['text'] for b in records)
                digest = hashlib.sha256(text.encode()).hexdigest()
                if any(b['canonical_page_sha256'] != digest or text[b['source_start']:b['source_end']] != b['text'] for b in records):
                    raise ValueError('Canonical page reconstruction failed')
                pages[page] = {'text': text, 'sha256': digest, 'blocks': records}
            self._pages[paper_id] = pages
        return self._pages[paper_id]

    def source(self, source_id):
        source = dict(self.sources[source_id])
        page = self.canonical_pages(source['paper_id'])[source['pdf_page']]
        if (source['document_sha256'] != self.papers[source['paper_id']]['document_sha256']
                or source['canonical_page_sha256'] != page['sha256']
                or page['text'][source['source_start']:source['source_end']] != source['text']):
            raise ValueError('Claim anchor does not match the original source')
        return self.decorate_source(source, 'claim_anchor')

    def decorate_source(self, source, origin):
        source = {**source, 'title': self.papers[source['paper_id']]['title'], 'origin': origin,
                  'source_validation': 'matched_current_pdf_and_canonical_text'}
        source['study_url'] = '/study?paper='+quote(source['paper_id'], safe='')+'&page='+str(source['pdf_page'])
        return source

    def hit_sources(self, hit):
        if hit['paper_id'] not in self.papers or hit['document_sha256'] != self.papers[hit['paper_id']]['document_sha256']:
            raise ValueError('Supplemental retrieval escaped the current graph corpus')
        page = self.canonical_pages(hit['paper_id'])[hit['pdf_page']]
        intervals = [(s['source_start'], s['source_end']) for s in hit.get('source_spans', [])]
        if intervals and any(hit['text'][s['text_start']:s['text_end']] != page['text'][s['source_start']:s['source_end']]
                             for s in hit['source_spans']):
            raise ValueError('Supplemental structural text differs from its source coordinates')
        if not intervals:
            start = page['text'].find(hit['text'].strip())
            if start < 0 or page['text'].find(hit['text'].strip(), start+1) >= 0:
                raise ValueError('Supplemental body does not map uniquely to its canonical page')
            intervals = [(start, start+len(hit['text'].strip()))]
        result = []
        for start, end in intervals:
            if not 0 <= start < end <= len(page['text']):
                raise ValueError('Supplemental source interval exceeds the canonical page')
            text = page['text'][start:end]
            records = [b for b in page['blocks'] if b['source_start'] < end and b['source_end'] > start]
            if not records or not text:
                raise ValueError('Invalid supplemental source interval')
            boxes = [b['bbox'] for b in records]
            source = {'source_id': 'E'+fingerprint([hit['document_sha256'],hit['pdf_page'],start,end])[:24],
                'paper_id': hit['paper_id'], 'document_sha256': hit['document_sha256'], 'pdf_page': hit['pdf_page'],
                'canonical_page_sha256': page['sha256'], 'source_start': start, 'source_end': end,
                'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'text': text,
                'bbox': [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]}
            result.append(self.decorate_source(source, 'supplemental_body'))
        return result

    def route_cells(self, topic_id):
        topic = next(g for g in self.communities['topic'] if g['community_id'] == topic_id)
        cells = []
        for method in self.communities['method']:
            members = sorted(set(topic['members']) & set(method['members']))
            if members:
                cells.append({'route_id': topic_id+'__'+method['community_id'], 'method_community_id': method['community_id'],
                    'topic_community_id': topic_id, 'paper_ids': members})
        return cells
