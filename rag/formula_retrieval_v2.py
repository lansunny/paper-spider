"""Rank source-linked formula candidates without new vectors or image rewriting."""
import re

from rag.config import ROOT
from rag.formula_retrieval import NUMBER_QUERY, attach_formulas
from rag.formula_store import FormulaStore

VERSION = 'source-join-formula-ranking-v2'


def attach_formulas_v2(question, evidence, *, paper_ids=None, documents=None, root=ROOT,
                       store=None, limit=12, models=None, facets=None, rank=True):
    if limit < 0:
        raise ValueError('Formula limit must be nonnegative')
    store = store or FormulaStore(root)
    # Collect first, rank second, cap last. Original crops/proxies are unchanged.
    result = attach_formulas(question, evidence, paper_ids=paper_ids, documents=documents,
                             root=root, store=store, limit=100000)
    formulas = result['formulas']
    diagnostic = result['formula_retrieval']
    numbers = {m.group(1) for m in NUMBER_QUERY.finditer(question)}
    if numbers:
        # Never infer an exact-number scope from whichever papers retrieval found.
        if len(set(paper_ids or [])) != 1:
            formulas = []
            diagnostic['diagnostics'].append({'reason': 'exact_number_needs_explicit_single_paper'})
        else:
            formulas = [f for f in formulas if f['equation_number'] in numbers and
                        any(m['method'] == 'exact_equation_number' for m in f['matches'])]
            if not formulas:
                diagnostic['diagnostics'].append({'reason': 'requested_number_not_found_or_ambiguous',
                                                  'numbers': sorted(numbers)})
    # A request to inspect code/table labels is not a request for nearby equations.
    nonmath_check = (re.search(r'代码|函数调用|方案标签|表格标签|\bcode\b|function call|table label', question, re.I)
                     and re.search(r'错误展示|误判|误报|incorrect|false positive|'
                                   r'(?:不是|不应(?:作为)?)(?:独立)?(?:数学)?公式|not.*formula', question, re.I))
    if nonmath_check:
        formulas = []
        diagnostic['diagnostics'].append({'reason': 'non_formula_inspection_intent'})
    scored = False
    score_pairs = 0
    if rank and models is not None and formulas:
        payloads = []
        by_id = {}
        for f in formulas:
            d = store.document(f['paper_id'], f['document_sha256'])
            e = next(e for e in d['elements'] if e.element_id == f['element_id'])
            # Source explanations provide meaning; the PDF text proxy is marked
            # untrusted and only helps ranking. It never becomes a model answer.
            context = '\n'.join(s.text for s in e.context)
            key = f['paper_id']+'/'+f['element_id']
            text = (f'Title: {e.title}\nSection: {e.section_title or ""}\n'
                    f'Source explanation: {context[:2200]}\nUnverified PDF formula text: {e.raw_text[:700]}')
            payloads.append({'chunk_id': key, 'paper_id': e.paper_id, 'text': text})
            by_id[key] = f
        ranked = models.rerank(question, payloads)
        score_pairs += len(payloads)
        # Cross-paper formulas are scored against their own question facet.
        for facet in facets or []:
            scoped = [h for h in payloads if h['paper_id'] == facet['paper_id']]
            if scoped:
                local = {h['chunk_id']: h['score'] for h in models.rerank(facet['question'], scoped)}
                score_pairs += len(scoped)
                ranked = [{**h, 'score': local.get(h['chunk_id'], h['score'])} for h in ranked]
        source_positions = {h['chunk_id']: i for i, h in enumerate(evidence)}

        def key(h):
            f = by_id[h['chunk_id']]
            exact = any(m['method'] == 'exact_equation_number' for m in f['matches'])
            earliest = min((source_positions.get(m['source_chunk_id'], 100000) for m in f['matches']), default=100000)
            return (-int(exact), -h['score'], earliest, f['paper_id'], f['pdf_page'], f['element_id'])

        ranked.sort(key=key)
        formulas = [{**by_id[h['chunk_id']], 'ranking': {'method': 'local_cross_encoder_source_proxy',
                     'score': h['score'], 'score_is_probability': False}} for h in ranked]
        # Reserve one display slot per explicitly named paper, if slots allow.
        requested = list(dict.fromkeys(f['paper_id'] for f in facets or []))
        front = []
        if len(requested) <= limit:
            for pid in requested:
                best = next((f for f in formulas if f['paper_id'] == pid), None)
                if best:
                    front.append(best)
        selected_ids = {(f['paper_id'], f['element_id']) for f in front}
        formulas = front+[f for f in formulas if (f['paper_id'], f['element_id']) not in selected_ids]
        scored = True
    kept = formulas[:limit]
    kept_ids = {(f['paper_id'], f['element_id']) for f in kept}
    for h in result['evidence']:
        h['formula_ids'] = [eid for eid in h['formula_ids'] if (h['paper_id'], eid) in kept_ids]
    result['formulas'] = kept
    diagnostic.update(version=VERSION, ranking_performed=scored,
                      rerank_pairs=score_pairs,
                      ranking_input='source title + section + explanation + untrusted PDF proxy' if scored else None,
                      eligible_after_intent_filter=len(formulas), returned=len(kept), limit=limit)
    return result


def rescue_formula_sources(question, candidates, facets, *, models, documents, root=ROOT, store=None):
    """Reserve up to two formula-bearing source chunks per requested paper.

    This ranks formulas ALREADY linked to recall candidates, then follows their
    existing match IDs back to the source chunks. No new retrieval or gold input.
    """
    if not facets or not re.search(r'公式|原式|方程|\bequations?\b|\bformulas?\b', question, re.I):
        return [], {'performed':False,'reason':'not_an_explicit_cross_paper_formula_request'}
    attached=attach_formulas_v2(question,candidates,documents=documents,root=root,store=store,
                               models=models,facets=facets,limit=100000)
    positions={h['chunk_id']:i for i,h in enumerate(candidates)}
    chosen=[]
    for facet in facets:
        count=0
        for f in attached['formulas']:
            if f['paper_id']!=facet['paper_id']:
                continue
            links=[m for m in f['matches'] if m['source_chunk_id'] in positions]
            links.sort(key=lambda m:(m['method']!='formula_inside_chunk',positions[m['source_chunk_id']]))
            if links and links[0]['source_chunk_id'] not in chosen:
                chosen.append(links[0]['source_chunk_id']);count+=1
                if count==2:
                    break
    return chosen, {'performed':True,'method':'rank_linked_formulas_then_reserve_source_chunks',
                    'source_chunk_ids':chosen,'formula_candidates':attached['formula_retrieval']['matched_before_limit'],
                    'rerank_pairs':attached['formula_retrieval']['rerank_pairs']}
