"""CJK bigrams and an explicit scientific term bridge for English source PDFs.

This is query expansion, not automatic translation. Source text stays immutable.
"""
import json
import re

from rank_bm25 import BM25Okapi

from rag.config import ROOT
from rag.retrieval import STOPWORDS

VERSION = 'cjk-bigrams-scientific-terms-v2'


def lexical_tokens(text):
    result = []
    for word in re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]+', text.lower()):
        if '\u4e00' <= word[0] <= '\u9fff':
            result.extend(word[i:i+2] for i in range(len(word)-1))
            if len(word) == 1:
                result.append(word)
        elif word not in STOPWORDS:
            result.append(word)
    return result


class LexicalIndexV2:
    def __init__(self, chunks, root=ROOT, *, expand=True):
        self.chunks, self.expand = chunks, expand
        self.glossary = json.loads((root/'configs/lexical_terms_zh_en.json').read_text(encoding='utf-8'))
        # Title/section help identify source context, but payload bodies are unchanged.
        corpus = [lexical_tokens(c.title+' '+c.section_title+' '+c.text) for c in chunks]
        self.index = BM25Okapi(corpus) if any(corpus) else None

    def query(self, question):
        used, occupied = [], set()
        lowered = question.lower()
        for term, en in sorted(self.glossary.items(), key=lambda p: -len(p[0])):
            for match in re.finditer(re.escape(term), lowered):
                positions = set(range(*match.span()))
                if not positions.intersection(occupied):
                    occupied.update(positions)
                    used.append((term, en))
        additions = ' '.join(en for _, en in used) if self.expand else ''
        # BM25 treats a query as a bag of tokens; repeated expansions must not
        # multiply the weight of generic words shared by many glossary entries.
        tokens = list(dict.fromkeys(lexical_tokens(question+' '+additions)))
        return {'tokens': tokens, 'expanded_terms': dict(used) if self.expand else {},
                'text': question+(' '+additions if additions else ''), 'version': VERSION}

    def search(self, question, top_k=30, paper_ids=None):
        if self.index is None:
            return []
        terms = self.query(question)['tokens']
        if not terms:
            return []
        scores = self.index.get_scores(terms)
        allowed = set(paper_ids or [])
        indices = [i for i, c in enumerate(self.chunks) if (not allowed or c.paper_id in allowed) and scores[i] > 0]
        indices.sort(key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))
        return [{**self.chunks[i].payload(), 'score': float(scores[i]), 'scores': {'bm25': float(scores[i])}}
                for i in indices[:top_k]]
