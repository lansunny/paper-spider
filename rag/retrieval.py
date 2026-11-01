"""Comparable BM25/dense retrieval plus rank-based fusion, with score provenance."""
import re

from rank_bm25 import BM25Okapi

STOPWORDS = set('a an the is are was were be been being do does did of to in on at and or for with by from as it its this that what how which'.split())


def lexical_tokens(text):
    return [word for word in re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text.lower()) if word not in STOPWORDS]


class LexicalIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        corpus = [lexical_tokens(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(corpus) if any(corpus) else None

    def search(self, question, top_k=30, paper_ids=None):
        if self.index is None:
            return []
        terms = lexical_tokens(question)
        if not terms:
            return []
        scores = self.index.get_scores(terms)
        allowed = set(paper_ids or [])
        indices = [i for i, chunk in enumerate(self.chunks) if (not allowed or chunk.paper_id in allowed) and scores[i] > 0]
        indices.sort(key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))
        return [{**self.chunks[i].payload(), 'score': float(scores[i]), 'scores': {'bm25': float(scores[i])}}
                for i in indices[:top_k]]


def reciprocal_rank_fusion(rankings, k=60):
    if k < 1:
        raise ValueError('RRF k must be positive.')
    merged = {}
    for name, hits in rankings.items():
        seen = set()
        for rank, hit in enumerate(hits, 1):
            key = hit['chunk_id']
            if key in seen:
                continue
            seen.add(key)
            if key not in merged:
                merged[key] = {**hit, 'score': 0., 'scores': {}, 'ranks': {}}
            merged[key]['score'] += 1 / (k + rank)
            merged[key]['scores'].update(hit.get('scores', {}))
            merged[key]['ranks'][name] = rank
    for hit in merged.values():
        hit['scores']['rrf'] = hit['score']
    return sorted(merged.values(), key=lambda hit: (-hit['score'], hit['chunk_id']))


def rerank_candidates(models, question, candidates, *, title=False):
    """Prefix titles only for scoring; keep exact source bodies in returned hits."""
    originals = {hit['chunk_id']: hit for hit in candidates}
    inputs = [{**hit, 'text': f"Title: {hit['title']}\nText: {hit['text']}"}
              for hit in candidates] if title else candidates
    ranked = models.rerank(question, inputs)
    return [{**hit, 'text': originals[hit['chunk_id']]['text']} for hit in ranked]


def retrieval_metrics(hits, annotations):
    eligible = [a for a in annotations if a.get('text_retrieval_eligible')]
    if not eligible:
        return None
    retrieved = {pid for hit in hits for pid in hit.get('paragraph_ids', [])}
    recalls = [sum(bool(set(group) & retrieved) for group in a['evidence_paragraph_groups']) /
               len(a['evidence_paragraph_groups']) for a in eligible]
    relevant = {pid for a in eligible for group in a['evidence_paragraph_groups'] for pid in group}
    mrr = next((1 / rank for rank, hit in enumerate(hits, 1)
                if relevant.intersection(hit.get('paragraph_ids', []))), 0.)
    return {'evidence_recall': max(recalls), 'mrr': mrr}
