"""Expose Docling paragraphs to the research workflow's retrieval contract."""
from rag.config import ROOT
from rag.query_planning_v2 import PaperRouterV2, paper_facets


class DoclingResearch:
    def __init__(self, engine):
        self.engine = engine
        self.root = ROOT
        self.settings = engine.cfg
        self.router = PaperRouterV2.from_project(engine.manifest['documents'], self.root)

    def snapshot(self):
        return self.engine.manifest, None

    def retrieve(self, question, *, mode='hybrid', rerank=True, paper_ids=None, auto_route=True):
        if paper_ids:
            route = {'mode': 'manual', 'paper_ids': list(paper_ids)}
        elif auto_route:
            route = self.router.route(question)
        else:
            route = {'mode': 'global', 'paper_ids': []}
        scope = route['paper_ids'] or None
        facets = paper_facets(question, self.router, allowed_ids=scope) if auto_route else []
        hits, counts = self.engine.retrieve_body(question, paper_ids=scope, facets=facets, mode=mode, rerank=rerank)
        documents = self.engine.manifest['documents']
        candidates = []
        for hit in hits:
            if hit['view'] != 'paragraph' or documents.get(hit['paper_id']) != hit['document_sha256']:
                raise RuntimeError('Docling 正文证据与当前论文版本不一致。')
            start, end = hit['view_span']
            pages = sorted({p['page_no'] for item in hit['elements'] for p in item['provenance']
                            if (span := p.get('charspan')) and start < span[1] and span[0] < end})
            candidates.append({
                **{key: hit[key] for key in ('paper_id', 'document_sha256', 'title', 'text', 'chunk_id',
                                            'object_id', 'view_span', 'score', 'section_title')},
                'pdf_page': pages[0] if pages else hit['pdf_page'],
                'pdf_pages': pages or [hit['pdf_page']],
            })
        return {'snapshot_id': self.engine.manifest['snapshot_id'], 'route': route, 'reranked_candidates': candidates,
                'candidate_counts': counts, 'retrieval_backend': 'docling_body'}
