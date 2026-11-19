"""Compare query languages for Docling body retrieval with fixed evidence and budgets."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
import unicodedata
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.docling_search import DoclingSearch
from rag.query_translation import HAN, translate_query
from rag.storage import atomic_json
from scripts.docling_prepare import digest
from scripts.evaluate_docling import DATA, load_set

LEGACY = ROOT/'data/evaluation/paper_benchmark/test50.jsonl'
MODES = ('original', 'bm25_english', 'english')


def normalized_char(char):
    return ''.join(c for c in unicodedata.normalize('NFKC', char).casefold() if c.isalnum())


def page_text(db, paper):
    pages = defaultdict(lambda: ([], []))
    for raw, in db.execute("SELECT payload FROM objects WHERE kind='paragraph' AND paper_id=? "
                           "ORDER BY ordinal", (paper,)):
        obj = json.loads(raw)
        for source in obj['provenance']:
            text, coords = pages[(obj['pdf_sha256'], source['page_no'])]
            start, end = source['charspan']
            for offset, char in enumerate(obj['text'][start:end], start):
                normalized = normalized_char(char)
                text.append(normalized)
                coords.extend([(obj['object_id'], offset)] * len(normalized))
    return {key: (''.join(text), coords) for key, (text, coords) in pages.items()}


def map_span(db, source, papers):
    paper = source['paper_id']
    if paper not in papers:
        papers[paper] = page_text(db, paper)
    page, coords = papers[paper].get((source['document_sha256'], source['pdf_page']), ('', []))
    quote = ''.join(map(normalized_char, source['quote']))
    begin = page.find(quote) if quote else -1
    if begin < 0 or page.find(quote, begin+1) >= 0:
        return []
    spans = []
    for oid, offset in coords[begin:begin+len(quote)]:
        if spans and spans[-1][0] == oid:
            spans[-1][2] = offset+1
        else:
            spans.append([oid, offset, offset+1])
    return spans


def legacy_cases(db):
    papers = {}
    selected, excluded = [], []
    with LEGACY.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['category'] != 'single_paper' or not row['answerable']:
                continue
            units = map_units(db, row, papers)
            if not units or not all(units):
                excluded.append(row['question_id'])
                continue
            selected.append({'id': row['question_id'], 'query': row['question'], 'cohort': 'legacy_single_paper',
                             'units': units})
    return selected, excluded


def map_units(db, question, papers):
    units = []
    for unit in question['evidence_units']:
        alternatives = []
        for source_group in unit['alternatives']:
            mapped = [map_span(db, span, papers) for span in source_group]
            if mapped and all(mapped):
                alternatives.append([span for part in mapped for span in part])
        units.append(alternatives)
    return units


def evidence_score(hits, units):
    windows = defaultdict(list)
    for hit in hits:
        windows[hit['object_id']].append(hit['view_span'])
    for intervals in windows.values():
        intervals.sort()

    def covered(span):
        oid, start, end = span
        cursor = start
        for left, right in windows[oid]:
            if left <= cursor < right:
                cursor = right
        return cursor >= end

    passed = [any(all(covered(span) for span in alternative) for alternative in unit) for unit in units]
    return {'any_evidence': any(passed), 'complete': all(passed), 'unit_coverage': mean(passed)}


def summarize(rows):
    summary = {}
    for label, items in [('all', rows), *[(label, [r for r in rows if r['cohort'] == label])
                                        for label in sorted({r['cohort'] for r in rows})]]:
        summary[label] = {'n': len(items), 'modes': {}, 'paired_complete_vs_original': {}}
        for mode in MODES:
            measures = [item['modes'][mode] for item in items]
            summary[label]['modes'][mode] = {k: mean(row[k] for row in measures)
                                             for k in ('any_evidence', 'complete', 'unit_coverage')}
            if mode != 'original':
                gained = [r['id'] for r in items if r['modes'][mode]['complete'] and not r['modes']['original']['complete']]
                lost = [r['id'] for r in items if r['modes']['original']['complete'] and not r['modes'][mode]['complete']]
                summary[label]['paired_complete_vs_original'][mode] = {'gained': gained, 'lost': lost}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='8bde7f9ad04def90')
    parser.add_argument('--test50-only', action='store_true', help='Run the eight original Docling body questions only')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    snapshot, cases = load_set(args.profile)
    engine = DoclingSearch(args.profile, body_rerank_k=20)
    if engine.manifest['object_snapshot_id'] != snapshot['snapshot_id'] or len(engine.manifest['documents']) != 500:
        raise RuntimeError('The published full text index must match the annotated snapshot')
    selected = [
        {'id': row['id'], 'query': row['query'], 'cohort': 'docling_body',
         'units': [[[[gold['object_id'], *gold['span']]]] for gold in row['gold']]}
        for row in cases if row['group'] == 'body'
    ]
    excluded = []
    if not args.test50_only:
        with closing(engine.connection()) as db:
            additional, excluded = legacy_cases(db)
        selected.extend(additional)
    if not all(HAN.search(case['query']) for case in selected):
        raise ValueError('Expected Chinese input questions in the translation comparison')
    target = ROOT/'runtime/experiments'/f'docling-query-language-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}'
    design = {'profile': args.profile, 'index': engine.manifest['snapshot_id'], 'object_snapshot': snapshot['snapshot_id'],
              'datasets': {path.name: digest(path) for path in ([DATA] if args.test50_only else [DATA, LEGACY])},
              'code_sha256': {path: digest(ROOT/path) for path in (
                  'scripts/evaluate_docling_query_language.py', 'rag/docling_search.py', 'rag/query_translation.py')},
              'top_k': 5, 'candidate_k': engine.cfg.candidate_k, 'rerank_k': engine.cfg.rerank_k,
              'unmapped_legacy_cases': excluded, 'cases': len(selected)}
    atomic_json(target/'design.json', design)
    print(f"Prepared {len(selected)} questions, {len(excluded)} unmapped legacy questions. {target}", flush=True)

    def timed_translation(question):
        started = perf_counter()
        value = translate_query(question)
        return value, round(perf_counter()-started, 3)

    scores, tokens, durations = [], [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {c['id']: executor.submit(timed_translation, c['query']) for c in selected}
        for case in selected:
            translated, seconds = pending[case['id']].result()
            if translated.tokens is not None:
                tokens.append(translated.tokens)
            durations.append(seconds)
            result = {**case, 'translation': translated.english, 'translation_model': translated.model,
                      'translation_seconds': seconds, 'translation_tokens': translated.tokens, 'modes': {}}
            for mode in MODES:
                started = perf_counter()
                found = engine.search(case['query'], kinds=('body',), top_k=5, query_language=mode)
                if mode != 'original' and found.get('query_translation', {}).get('status') != 'translated':
                    raise RuntimeError(f"Translation was unavailable for {case['id']}/{mode}")
                hits = [{k: hit[k] for k in ('object_id','chunk_id','paper_id','view_span')} for hit in found['body']]
                result['modes'][mode] = {**evidence_score(hits, case['units']), 'hits': hits,
                                         'elapsed_seconds': round(perf_counter()-started, 3)}
            atomic_json(target/f"{case['id']}.json", result)
            scores.append(result)
            print(case['id'], *(result['modes'][mode]['complete'] for mode in MODES), flush=True)
    atomic_json(target/'summary.json', {'design': design, 'groups': summarize(scores),
                                       'translation_median_seconds': median(durations),
                                       'translation_mean_tokens': mean(tokens) if tokens else None,
                                       'translation_models': sorted({r['translation_model'] for r in scores})})
    print(f'Results: {target / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
