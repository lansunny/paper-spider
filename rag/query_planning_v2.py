"""Conservative, question-only per-paper facets; no model calls or gold labels."""
import re
from dataclasses import asdict, dataclass

from rag.query_decomposition import allocate_slots
from rag.routing import PaperRouter, normalize

VERSION = 'explicit-paper-facets-v2'


class PaperRouterV2(PaperRouter):
    def mentions(self, question):
        text = normalize(question)
        found = [(m.start(), m.end(), term, ids) for term, ids, pattern in self.patterns
                 for m in pattern.finditer(text)]
        return [m for m in sorted(found, key=lambda m: (m[0], -m[1])) if not any(
            n[0] <= m[0] and n[1] >= m[1] and n[1]-n[0] > m[1]-m[0] for n in found)]

    def route(self, question):
        # Contrastive learning is an intra-paper topic, not a comparison intent.
        text = re.sub(r'对比(?=学习|损失|目标|预训练|训练)', '对照', question)
        result = super().route(text)
        result['router_version'] = VERSION
        return result


@dataclass(frozen=True)
class PaperFacet:
    question: str
    paper_id: str

    def record(self):
        return asdict(self)


def paper_facets(question, router, *, allowed_ids=None, max_papers=3):
    """Split explicit, unambiguous entities, preserving local user descriptions.

    An ambiguous acronym, unknown linked subject, or >max_papers is left global.
    Full titles come only from the already matched catalog entities.
    """
    mentions = router.mentions(question)
    if not mentions or any(len(m[3]) != 1 for m in mentions):
        return []
    ids = list(dict.fromkeys(next(iter(m[3])) for m in mentions))
    allowed = router.indexed if allowed_ids is None else set(allowed_ids)
    if not 2 <= len(ids) <= max_papers or not set(ids).issubset(allowed):
        return []
    text = normalize(question)
    # Quoted unresolved titles must not silently become one of the known papers.
    remainder = list(text)
    for a, z, _, _ in mentions:
        remainder[a:z] = ' '*(z-a)
    if re.search(r'[《“「][^》”」]*\S[^》”」]*[》”」]', ''.join(remainder)):
        return []
    last_tail = text[mentions[-1][1]:].strip(' 《》“”「」')
    requests = re.findall(r'(?:请|\bshow\b|\bexplain\b)[^。；;！？!?]*', last_tail, re.I)
    request = requests[-1] if requests else ''
    if not request:
        request = '请说明相关定义和方法。'
    request = re.sub(r'分别|两篇论文的?|两处|both papers?', '', request, flags=re.I)
    facets = []
    for pid in ids:
        i = next(i for i, m in enumerate(mentions) if pid in m[3])
        end = mentions[i+1][0] if i+1 < len(mentions) else len(text)
        local = text[mentions[i][1]:end].strip(' 《》“”「」')
        local = re.sub(r'(?:与|和|以及|及|而|versus|vs\.?|and|with|,|，|《|“|「)+\s*$', '', local, flags=re.I).strip()
        # Shared topic follows the last entity in "A 与 B 的训练损失".
        if not local or local in {'的', '论文'}:
            local = last_tail
        local = re.sub(r'分别|两篇论文的?|两处|both papers?', '', local, flags=re.I)
        facet = f'《{router.titles[pid]}》 {local}'
        if request not in local:
            facet += '；'+request
        facets.append(PaperFacet(facet, pid))
    return facets


def balanced_candidates(original, facets, fused, limit):
    """Keep a fixed total budget, reserving half for scoped facets before dedup."""
    if limit < 1:
        raise ValueError('Candidate limit must be positive')
    pools = [original, *facets]
    if len(pools) > limit:
        raise ValueError('Not enough candidate slots for all facets')
    quotas = allocate_slots(limit, len(pools))
    selected, seen = [], set()
    for pool, quota in zip(pools, quotas):
        added = 0
        for h in pool:
            if h['chunk_id'] not in seen:
                selected.append(h)
                seen.add(h['chunk_id'])
                added += 1
                if added == quota:
                    break
    for h in fused:
        if len(selected) >= limit:
            break
        if h['chunk_id'] not in seen:
            selected.append(h)
            seen.add(h['chunk_id'])
    return selected
