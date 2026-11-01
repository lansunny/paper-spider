"""Text-chunk -> source-interval -> formula join, without formula embeddings."""
import re
from collections import Counter
from urllib.parse import quote

from rag.config import ROOT
from rag.formula_store import FormulaStore


def overlap(a, z, b, y):
    return a < y and b < z  # half-open intervals; touching endpoints do not overlap


def aligned_spans(hit, page):
    """Validate explicit spans or recover a unique exact legacy-text occurrence."""
    canonical, text = page['text'], hit['text']
    if hit.get('source_spans'):
        intervals = []
        blocks = {b['number']: b for b in page['blocks']}
        for s in hit['source_spans']:
            a, z, x, y = (s[k] for k in ['source_start', 'source_end', 'text_start', 'text_end'])
            block = blocks.get(s['block_number'])
            if (not block or not block['start'] <= a < z <= block['end'] or not 0 <= x < y <= len(text)
                    or z-a != y-x or canonical[a:z] != text[x:y]):
                return [], 'source_coordinate_mismatch'
            intervals.append((a, z))
        return intervals, 'chunk_source_spans'
    at = canonical.find(text)
    if at < 0 or canonical.find(text, at+1) >= 0:
        return [], 'legacy_text_missing_or_ambiguous'
    return [(at, at+len(text))], 'legacy_unique_text_alignment'


def presentation(element, matches):
    return {'element_id': element.element_id, 'paper_id': element.paper_id,
            'document_sha256': element.document_sha256, 'title': element.title,
            'pdf_page': element.pdf_page, 'equation_number': element.equation_number,
            'bbox': element.bbox, 'crop_bbox': element.crop_bbox, 'crop_scope': element.crop_scope,
            'image_url': '/api/formulas/'+quote(element.paper_id, safe='')+'/'+quote(element.element_id, safe='')+'/image',
            'image_sha256': element.image_sha256, 'text_proxy': element.search_text,
            'proxy_trust': element.proxy_trust, 'proxy_for_retrieval_only': True,
            'model_has_seen_image': False, 'detection_confidence_band': element.confidence_band,
            'review_status': element.review_status, 'matches': matches}


NUMBER_QUERY = re.compile(r'(?:eq(?:uation)?\.?|formula|公式|方程|式)\s*[（(]?\s*((?:[A-Z]\.?\d+|\d+)(?:\.\d+)*[a-h]?)\s*[）)]?', re.I)


def symbolic_key(text):
    return re.sub(r'\s+', '', text).replace('−', '-').replace('≥', '>=').replace('≤', '<=')


def attach_formulas(question, evidence, *, paper_ids=None, documents=None, root=ROOT, store=None, limit=12):
    """Images supplement UI evidence; proxies never enter the generation prompt.

    A coordinate-body match and a neighboring explanation have different labels.
    Exact equation numbers require one paper; symbols are only a lexical fallback.
    """
    store = store or FormulaStore(root)
    enriched = [{**h, 'formula_ids': []} for h in evidence]
    matches, elements, diagnostics = {}, {}, []

    def add(e, method, hit=None, alignment=None):
        key = (e.paper_id, e.element_id)
        elements[key] = e
        value = {'method': method, 'source_chunk_id': hit.get('chunk_id') if hit else None}
        if alignment:
            value['alignment'] = alignment
        if value not in matches.setdefault(key, []):
            matches[key].append(value)
        if hit is not None and e.element_id not in hit['formula_ids']:
            hit['formula_ids'].append(e.element_id)

    for hit in enriched:
        d = store.document(hit['paper_id'], hit['document_sha256'])
        if d is None:
            diagnostics.append({'paper_id': hit['paper_id'], 'reason': 'no_matching_authoritative_formula_artifact'})
            continue
        page = d['pages'].get(hit.get('pdf_page'))
        if not page:
            continue
        spans, alignment = aligned_spans(hit, page)
        if not spans:
            diagnostics.append({'chunk_id': hit.get('chunk_id'), 'reason': alignment})
            continue
        numbers = Counter(e.equation_number for e in d['elements'] if e.equation_number)
        for e in d['elements']:
            for sources, method in [(e.source_spans, 'formula_inside_chunk'),
                                    (e.references if numbers[e.equation_number] == 1 else [], 'explicit_formula_reference'),
                                    (e.context, 'adjacent_explanation')]:
                if any(s.pdf_page == hit['pdf_page'] and any(overlap(a, z, s.source_start, s.source_end) for a, z in spans)
                       for s in sources):
                    add(e, method, hit, alignment)
                    break

    scope = list(dict.fromkeys(paper_ids or [h['paper_id'] for h in evidence]))
    numbers = list(dict.fromkeys(m.group(1) for m in NUMBER_QUERY.finditer(question)))
    # Bare '(1)' is not a query. Never choose Eq.1 arbitrarily across papers.
    if numbers and len(scope) == 1:
        pid = scope[0]
        d = store.document(pid, documents.get(pid) if documents else None)
        if d:
            for number in numbers:
                choices = [e for e in d['elements'] if e.equation_number == number]
                if len(choices) == 1:
                    add(choices[0], 'exact_equation_number')
                elif len(choices) > 1:
                    diagnostics.append({'paper_id': pid, 'reason': 'ambiguous_equation_number', 'number': number})
    elif numbers:
        diagnostics.append({'reason': 'equation_number_requires_single_paper_scope'})
    # Symbol fallback only handles literal source symbols, not invented LaTeX.
    symbolic = re.findall(r'[A-Za-z\u0370-\u03ff][A-Za-z\u0370-\u03ff0-9_]*(?:\([^)]{1,25}\))?\s*(?:=|≥|≤|≈|∈)\s*'
                          r'[A-Za-z\u0370-\u03ff0-9_+*/∑∫∏√∞×÷|^{}()\[\].,<>=≈≠≤≥∈ \-−]{1,55}', question)
    for pid in scope:
        d = store.document(pid, documents.get(pid) if documents else None)
        if d:
            for e in d['elements']:
                if any(symbolic_key(s) in symbolic_key(e.raw_text) for s in symbolic):
                    add(e, 'literal_symbol_proxy_fallback')
    ordered = list(elements)[:limit]
    kept = set(ordered)
    for h in enriched:
        h['formula_ids'] = [eid for eid in h['formula_ids'] if (h['paper_id'], eid) in kept]
    return {'evidence': enriched, 'formulas': [presentation(elements[k], matches[k]) for k in ordered],
            'formula_retrieval': {'method': 'source_span_join_plus_number_and_symbol', 'vector_index': False,
                                  'matched_before_limit': len(elements), 'returned': len(ordered),
                                  'diagnostics': diagnostics, 'ocr_performed': False,
                                  'generation_uses_formula_images': False}}
