"""Run the original 50 text questions on the full Docling index."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.docling_search import BODY_RERANK_DEPTH, DoclingSearch
from rag.generation import select_context
from rag.ingestion import fingerprint
from rag.query_translation import HAN, Translation, translate_query
from rag.storage import atomic_json, read_json
from scripts.accept_heldout import STAGES
from scripts.accept_test50 import question_hash, validate_cohorts
from scripts.docling_prepare import digest
from scripts.evaluate_docling_query_language import LEGACY, evidence_score, map_units

BASELINE_BUDGET = 8000


def brief(hit):
    return {key: hit[key] for key in ('chunk_id', 'paper_id', 'object_id', 'pdf_page', 'view_span', 'score')}


def historical_results(engine, questions, dataset_hash):
    acceptance = ROOT/'runtime/acceptance'
    frozen = read_json(acceptance/'frozen_baseline.json')
    if frozen is None:
        return None, {}
    spec = frozen['spec']
    target = acceptance/f"test50-{frozen['spec_sha256'][:12]}-{dataset_hash[:8]}"
    run = read_json(target/'run.json')
    if not run or run['status'] != 'completed' or run['dataset_sha256'] != dataset_hash:
        return None, {}
    if run['spec_sha256'] != frozen['spec_sha256'] or spec['documents_digest'] != fingerprint(engine.manifest['documents']):
        raise ValueError('The historical run and Docling must use the same benchmark and PDF versions')
    results = {q['question_id']: read_json(target/f"{q['question_id']}.json") for q in questions}
    if any(not results[q['question_id']] or results[q['question_id']]['question_sha256'] != question_hash(q)
           for q in questions):
        raise ValueError('Historical acceptance rows do not match test50')
    return {'path': str(target.relative_to(ROOT)), 'snapshot': spec['snapshot_id'],
            'settings': spec['settings'], 'routing': spec['routing'],
            'lexical_policy': spec['lexical_policy_for_acceptance']}, results


def summarize(rows):
    stages = ('union', 'rrf20', 'after_top5', 'context_budget', 'context_native')
    report = {'questions': len(rows), 'docling': {}, 'historical': {}}
    if not rows:
        return report
    for source in ('docling', 'historical'):
        if source not in rows[0]:
            continue
        for stage in stages:
            if stage not in rows[0][source]:
                continue
            metrics = [r[source][stage] for r in rows]
            report[source][stage] = {
                'evidence_coverage': mean(m['evidence_unit_recall'] for m in metrics),
                'complete_evidence_count': sum(m['all_required_evidence'] for m in metrics),
            }
    if report['historical']:
        for stage in STAGES:
            now = {r['question_id'] for r in rows if r['docling'][stage]['all_required_evidence']}
            old = {r['question_id'] for r in rows if r['historical'][stage]['all_required_evidence']}
            report['docling'][stage].update(gained=sorted(now-old), lost=sorted(old-now))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='8bde7f9ad04def90')
    parser.add_argument('--body-rerank-k', type=int, default=BODY_RERANK_DEPTH, choices=range(1, 201),
                        metavar='1..200')
    parser.add_argument('--translations-from', type=Path, help='Reuse recorded model translations from a completed run')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    dataset_hash = digest(LEGACY)
    metadata = read_json(LEGACY.with_name('test50_manifest.json'))
    if dataset_hash != metadata['dataset_sha256']:
        raise ValueError('Original test50 file differs from its manifest')
    with LEGACY.open(encoding='utf-8') as stream:
        questions = [json.loads(line) for line in stream if line.strip()]
    validate_cohorts(questions, metadata)
    engine = DoclingSearch(args.profile, body_rerank_k=args.body_rerank_k)
    if len(engine.manifest['documents']) != 500:
        raise ValueError('Run the benchmark against the published 500-paper index')
    for q in questions:
        for unit in q['evidence_units']:
            for alternative in unit['alternatives']:
                for span in alternative:
                    if engine.manifest['documents'].get(span['paper_id']) != span['document_sha256']:
                        raise ValueError('A gold PDF version differs from the Docling index')

    baseline, historical = historical_results(engine, questions, dataset_hash)
    matched_budget = baseline['settings']['context_char_budget'] if baseline else BASELINE_BUDGET
    native_budget = engine.cfg.context_char_budget
    release = engine.manifest['snapshot_id']
    replayed = {}
    if args.translations_from:
        previous = read_json(args.translations_from/'summary.json')
        if (not previous or previous['cases_run'] != len(questions)
                or previous['design']['dataset_sha256'] != dataset_hash or previous['design']['index'] != release):
            raise ValueError('Translation source must be a completed run on the same test50 and index')
        for q in questions:
            row = read_json(args.translations_from/f"{q['question_id']}.json")
            if not row or row['question'] != q['question']:
                raise ValueError('Recorded translation does not match the benchmark question')
            if HAN.search(q['question']):
                if not row['translation'] or HAN.search(row['translation']):
                    raise ValueError('Expected a recorded English model translation')
                replayed[q['question_id']] = Translation(row['translation'], row['translation_model'], row['translation_tokens'])
    target = ROOT/'runtime/experiments'/f'docling-text50-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}'
    design = {'dataset_sha256': dataset_hash, 'questions': len(questions), 'profile': args.profile, 'index': release,
              'object_snapshot': engine.manifest['object_snapshot_id'], 'historic_run': baseline,
              'query_language': 'english by default', 'question_input': ('replayed English model translations' if replayed
                                                                         else 'original Chinese; no benchmark translation or gold'),
              'translations_from': str(args.translations_from) if args.translations_from else None,
              'paper_filter': None, 'candidate_k': engine.cfg.candidate_k, 'rerank_k': engine.body_rerank_k,
              'rrf_k': engine.cfg.rrf_k, 'top_k': 5, 'matched_context_chars': matched_budget,
              'native_context_chars': native_budget, 'generation_api': False,
              'code_sha256': {name: digest(ROOT/name) for name in (
                  'rag/docling_search.py','rag/query_translation.py','scripts/evaluate_docling_text50.py',
                  'scripts/evaluate_docling_query_language.py')}}
    atomic_json(target/'design.json', design)
    print(f'Running all 50 questions: {target}', flush=True)

    def translate(question):
        started = perf_counter()
        value = translate_query(question)
        return value, round(perf_counter()-started, 3)

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        translations = {q['question_id']: pool.submit(translate, q['question'])
                        for q in questions if not replayed and HAN.search(q['question'])}
        pages = {}
        with closing(engine.connection()) as db:
            mappings = {q['question_id']: map_units(db, q, pages) for q in questions if q['answerable']}
        wanted = {span[0] for units in mappings.values() for unit in units for alt in unit for span in alt}
        windows = [{'object_id': c.object_id, 'view_span': c.view_span}
                   for c in engine.lexical['body'].chunks if c.object_id in wanted]
        index_ceilings = {key: [evidence_score(windows, [unit])['complete'] for unit in units]
                          for key, units in mappings.items()}
        for number, q in enumerate(questions, 1):
            qid = q['question_id']
            translation, duration = translations[qid].result() if qid in translations else (replayed.get(qid), None)
            started = perf_counter()
            query = replayed[qid].english if qid in replayed else q['question']
            found = engine.search(query, kinds=('body',), top_k=5, trace=True)
            if found['snapshot_id'] != release or (qid in translations and found.get('query_translation', {}).get('status') != 'translated'):
                raise RuntimeError(f'{qid} did not use the expected snapshot and English retrieval')
            trace = found['rankings']['body']
            small, small_chars = select_context(trace['reranked'], matched_budget)
            by_chunk = {hit['chunk_id']: hit for hit in trace['reranked']}
            native = [by_chunk[hit['chunk_id']] for hit in found['body_context']]
            native_chars = found['body_context_chars']
            ranked = {'union': trace['union'], 'rrf20': trace['rrf20'], 'after_top5': trace['reranked'][:5],
                      'context_budget': small, 'context_native': native}
            result = {'question_id': qid, 'category': q['category'], 'scope': q['source_scope'],
                      'question': q['question'], 'answerable': q['answerable'],
                      'translation': translation.english if translation else None,
                      'translation_model': translation.model if translation else None, 'translation_seconds': duration,
                      'translation_tokens': translation.tokens if translation else None,
                      'retrieval_seconds': round(perf_counter()-started, 3),
                      'context_chars': {'matched': small_chars, 'native': native_chars},
                      'ranked': {stage: [brief(h) for h in hits] for stage, hits in ranked.items()}}
            if qid in historical:
                result['historical'] = historical[qid]['metrics']
                result['historical_route'] = historical[qid]['retrieval']['route']
            if q['answerable']:
                units = mappings[qid]
                result['missing_body_units'] = [unit['unit_id'] for unit, indexed in zip(q['evidence_units'], index_ceilings[qid], strict=True)
                                                if not indexed]
                result['mapped_units'] = units
                result['docling'] = {}
                for stage, hits in ranked.items():
                    score = evidence_score(hits, units)
                    result['docling'][stage] = {'evidence_unit_recall': score['unit_coverage'],
                                                'all_required_evidence': score['complete']}
            else:
                result['control'] = 'retrieval recorded; no positive gold or answer generated'
            atomic_json(target/f'{qid}.json', result)
            rows.append(result)
            status = f"top5 {result['docling']['after_top5']['evidence_unit_recall']:.1%}" if q['answerable'] else 'control'
            print(f'[{number}/50] {qid}: {status}', flush=True)

    answerable = [r for r in rows if r['answerable']]
    aligned = [r for r in answerable if not r['missing_body_units']]
    groups = {'all_answerable_lower_bound': summarize(answerable), 'fully_aligned': summarize(aligned),
              **{category: summarize([r for r in answerable if r['category'] == category])
                 for category in ('single_paper', 'cross_paper')},
              **{cohort: summarize([r for r in answerable if r['question_id'] in ids])
                 for cohort, ids in metadata['cohorts'].items()}}
    translation_times = [r['translation_seconds'] for r in rows if r['translation_seconds'] is not None]
    atomic_json(target/'summary.json', {'design': design, 'cases_run': len(rows), 'groups': groups,
        'body_mapping_gaps': {r['question_id']: r['missing_body_units'] for r in answerable if r['missing_body_units']},
        'controls': {r['question_id']: len(r['ranked']['after_top5']) for r in rows if not r['answerable']},
        'translation_models': sorted({r['translation_model'] for r in rows if r['translation_model']}),
        'translation_median_seconds': median(translation_times) if translation_times else None,
        'median_retrieval_seconds': median(r['retrieval_seconds'] for r in rows)})
    print(f'Results: {target / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
