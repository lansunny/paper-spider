"""Conservative question-only scope routing; never accepts benchmark labels."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from rag.config import ROOT
from rag.ingestion import paper_catalog


VERSION = 'explicit-single-paper-v1'
GLOBAL_INTENT = re.compile(
    r'对比|比较|相比|区别|异同|不同|优于|其他|其它|别的|类似|替代|有没有|'
    r'哪些论文|哪些研究|哪些方法|有哪些.*(?:论文|研究|方法)|相关工作|研究现状|'
    r'综述|推荐|最新|相关论文|跨论文|跨文献|'
    r'\b(?:compar\w*|versus|vs\.?|differences?|other|similar|survey|recommend\w*|latest|alternatives?)\b',
    re.I)
# Deliberately prefer global retrieval when a second, unresolved entity may exist.
LINK_AFTER = re.compile(r'^\s*(?:(?:原论文|论文|方法|系统|模型)\s*)?(?:与|和|以及|及|、|&|\b(?:and|with)\b)', re.I)
LINK_BEFORE = re.compile(r'(?:与|和|以及|及|、|&|\b(?:and|both)\b)\s*$', re.I)


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).translate(
        str.maketrans({'–': '-', '—': '-', '‑': '-'}))).strip().casefold()


class PaperRouter:
    def __init__(self, catalog, indexed_ids, aliases=None):
        self.indexed = set(indexed_ids)
        self.titles = {p['paper_id']: p['title'] for p in catalog}
        terms = {}

        def add(term, pid):
            term = normalize(term)
            if term:
                terms.setdefault(term, set()).add(pid)

        for pid, title in self.titles.items():
            add(title, pid)
            add(pid, pid)
            prefix = re.split(r'[:：]', title, maxsplit=1)
            if (len(prefix) == 2 and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{2,30}', prefix[0])
                    and sum(c.isupper() for c in prefix[0]) >= 2
                    and prefix[0].upper() not in {'RAG', 'LLM', 'NLP', 'BERT', 'GPT'}):
                add(prefix[0], pid)
        for pid, names in (aliases or {}).items():
            if pid not in self.titles:
                continue
            for name in names:
                add(name, pid)
        self.patterns = [(term, ids, re.compile(r'(?<![a-z0-9])'+re.escape(term)+r'(?![a-z0-9])'))
                         for term, ids in terms.items()]
        self.fingerprint = hashlib.sha256(json.dumps(
            {'titles': self.titles, 'terms': {k: sorted(v) for k, v in terms.items()},
             'indexed_ids': sorted(self.indexed)}, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_project(cls, indexed_ids, root=ROOT):
        alias_path = Path(root)/'configs/paper_aliases.json'
        aliases = json.loads(alias_path.read_text(encoding='utf-8')) if alias_path.exists() else {}
        return cls(paper_catalog(root), indexed_ids, aliases)

    def route(self, question):
        text = normalize(question)
        matches = [(m.start(), m.end(), term, ids) for term, ids, pattern in self.patterns
                   for m in pattern.finditer(text)]
        # A complete title can contain another paper's title or acronym. Retain
        # the longest containing mention; equal-span collisions remain ambiguous.
        matches = [m for m in matches if not any(
            n[0] <= m[0] and n[1] >= m[1] and n[1]-n[0] > m[1]-m[0] for n in matches)]
        ids = sorted({pid for m in matches for pid in m[3]})
        remainder = list(text)
        for start, end, _, _ in matches:
            remainder[start:end] = ' '*(end-start)
        remainder = ''.join(remainder)
        remainder = re.sub(r'[《“"「]\s*[》”"」]', ' ', remainder)
        linked = any(LINK_AFTER.search(text[end:]) or LINK_BEFORE.search(text[:start])
                     for start, end, _, _ in matches)
        decision = {'mode': 'global', 'paper_ids': [], 'matched_paper_ids': ids,
                    'matched_terms': sorted({m[2] for m in matches}),
                    'router_version': VERSION, 'catalog_fingerprint': self.fingerprint}
        if len(ids) > 1:
            reason = 'multiple_or_ambiguous_papers'
        elif GLOBAL_INTENT.search(remainder):
            reason = 'comparison_or_open_intent'
        elif not ids:
            reason = 'no_explicit_resolved_paper'
        elif ids[0] not in self.indexed:
            reason = 'paper_not_in_snapshot'
        elif linked or re.search(r'[《》“”"「」]', remainder):
            reason = 'possible_additional_subject'
        else:
            decision.update(mode='single_paper', paper_ids=ids)
            reason = 'unique_explicit_paper'
        decision['reason'] = reason
        return decision
