"""Run the existing text test50 against ordinary retrieval and Deep Research."""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.rag_service import get_pipeline
from rag.benchmark import load_source_pages, score_evidence
from rag.deep_research import research
from rag.generation import generation_settings
from rag.pipeline_v2 import PipelineV2
from rag.storage import atomic_json, read_json
from scripts.accept_test50 import validate_cohorts


RETRIEVAL_STAGES = ('ordinary_context', 'research_initial', 'research_final')
STAGES = (*RETRIEVAL_STAGES, 'research_cited_context')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scored(question, chunks, pages):
    return score_evidence(question, chunks, pages, k=max(1, len(chunks)))


def covered(score):
    return {unit['unit_id'] for unit in score['units'] if unit['covered']}


def summarize_group(rows):
    if not rows:
        return None
    trial = [row for row in rows if row['research']['generation_status'] == 'completed']
    result = {'questions': len(rows), 'with_followup': sum(bool(row['followups']) for row in rows),
              'research_cited_answer': sum(bool(row['research']['citations']) for row in rows),
              'research_followup_cited': sum(row['followup_cited'] for row in rows),
              'research_generation_status': dict(Counter(row['research']['generation_status'] for row in rows)),
              'research_stop_reason': dict(Counter(row['research']['stop_reason'] for row in rows)),
              'research_retrieval_median_ms': median(row['research']['timings']['retrieval_ms'] for row in rows),
              'ordinary_retrieval_median_ms': median(row['ordinary']['retrieval_ms'] for row in rows),
              'research_total_median_ms': median(row['research']['elapsed_ms'] for row in rows),
              'research_tokens_total': sum(row['research']['usage'].get('total_tokens', 0) for row in rows),
              'research_tokens_mean': mean(row['research']['usage'].get('total_tokens', 0) for row in rows),
              'research_completed_with_missing': sum(bool(row['research']['missing_modules']) for row in trial)}
    if not rows[0]['answerable']:
        result['controls'] = [{k: row[k] for k in ('question_id', 'ordinary', 'research')} for row in rows]
        return result
    result['evidence'] = {stage: {
        'mean_coverage': mean(row['metrics'][stage]['evidence_unit_recall'] for row in rows),
        'complete_count': sum(row['metrics'][stage]['all_required_evidence'] for row in rows),
        'complete_rate': mean(row['metrics'][stage]['all_required_evidence'] for row in rows),
    } for stage in STAGES if stage in rows[0]['metrics']}
    deltas = [row['metrics']['research_final']['evidence_unit_recall'] -
              row['metrics']['ordinary_context']['evidence_unit_recall'] for row in rows]
    result['research_vs_ordinary'] = {'mean_delta': mean(deltas), 'wins': sum(d > 0 for d in deltas),
                                       'losses': sum(d < 0 for d in deltas), 'ties': sum(d == 0 for d in deltas)}
    result['followup_added_units'] = sum(len(row['new_units']) for row in rows)
    result['followup_lost_units'] = sum(len(row['lost_units']) for row in rows)
    result['followup_improved_questions'] = sum(bool(row['new_units']) for row in rows)
    result['followup_completed_questions'] = sum(row['metrics']['research_final']['all_required_evidence'] and
                                                 not row['metrics']['research_initial']['all_required_evidence'] for row in rows)
    return result


def run_case(pipeline, question, pages, snapshot, settings, *, retrieval_only=False):
    # Only the original user question reaches either retrieval path.
    prompt = question['question']
    baseline = pipeline.ask(prompt, generate_answer=not retrieval_only and not question['answerable'])
    agent = research(pipeline, prompt, settings=settings, synthesize=not retrieval_only)
    if baseline['snapshot_id'] != snapshot or agent['snapshot_id'] != snapshot:
        raise ValueError('Index snapshot changed during evaluation')
    initial = agent['initial_evidence']
    cited = [hit for hit in agent['evidence'] if hit['citation_id'] in agent['citations']]
    evidence = [baseline['evidence'], initial, agent['evidence'], cited]
    stages = RETRIEVAL_STAGES if retrieval_only else STAGES
    scores = {stage: scored(question, hits, pages) for stage, hits in zip(stages, evidence)}
    ordinary = {key: baseline.get(key) for key in ('trace_id', 'generation_status', 'answer', 'citations',
                                                  'insufficient_evidence', 'usage', 'elapsed_ms', 'context_chars')}
    ordinary['retrieval_ms'] = max(0, baseline['elapsed_ms'] - baseline['timings'].get('generation_ms', 0))
    details = {key: agent[key] for key in ('trace_id', 'generation_status', 'answer', 'citations', 'missing_modules',
                                          'insufficient_evidence', 'usage', 'elapsed_ms', 'timings', 'context_chars')}
    details['stop_reason'] = agent['research']['stop_reason']
    return {'question_id': question['question_id'], 'question': prompt, 'answerable': question['answerable'],
            'category': question['category'], 'source_scope': question['source_scope'], 'metrics': scores,
            'ordinary': ordinary, 'research': details,
            'model_calls': agent['research']['model_calls'],
            'followups': [s for s in agent['research']['searches'] if s['round'] == 2],
            'followup_cited': any(hit['research_origin'] == 'followup' for hit in cited),
            'new_units': sorted(covered(scores['research_final']) - covered(scores['research_initial'])),
            'lost_units': sorted(covered(scores['research_initial']) - covered(scores['research_final']))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--retrieval-only', action='store_true',
                        help='Use real assessment and follow-up on answerable cases; skip synthesis.')
    parser.add_argument('--category', choices=['single_paper', 'cross_paper'], help='Evaluate one test50 cohort.')
    args = parser.parse_args()
    dataset = ROOT / 'data/evaluation/paper_benchmark/test50.jsonl'
    metadata = read_json(dataset.with_name('test50_manifest.json'))
    digest = sha(dataset)
    if digest != metadata['dataset_sha256']:
        raise ValueError('test50 does not match its manifest')
    questions = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines() if line]
    validate_cohorts(questions, metadata)
    if args.retrieval_only:
        questions = [q for q in questions if q['answerable']]
    if args.category:
        questions = [q for q in questions if q['category'] == args.category]
    pipeline = get_pipeline()
    if not isinstance(pipeline, PipelineV2):
        raise ValueError('V2 retrieval is required for this comparison')
    with closing(pipeline.store):
        settings = generation_settings(pipeline.root)
        if not settings.configured:
            raise ValueError('DeepSeek API is required for this evaluation')
        manifest, _ = pipeline.snapshot()
        snapshot = manifest['snapshot_id']
        pipeline.store.validate_collection()
        if pipeline.store.count(snapshot) != manifest['chunks']:
            raise ValueError('Qdrant does not contain the complete indexed snapshot')
        pages = load_source_pages({pid for q in questions for pid in q['gold_paper_ids']})
        if any(manifest['documents'].get(paper) != digest for paper, digest, _ in pages):
            raise ValueError('Gold papers differ from indexed PDF versions')
        design = {'test_sha256': digest, 'snapshot_id': snapshot, 'papers': len(manifest['documents']),
                  'chunks': manifest['chunks'], 'retrieval_settings': pipeline.settings.model_dump(),
                  'model': settings.model, 'max_output_tokens': settings.max_tokens,
                  'retrieval_only': args.retrieval_only, 'category': args.category,
                  'methods': {'ordinary': ('current V2, generation disabled' if args.retrieval_only else
                                           'current V2, generation only on four unsupported controls'),
                              'research': ('current Deep Research, no selected papers, actual assessment API; synthesis disabled'
                                           if args.retrieval_only else
                                           'current Deep Research, no selected papers, actual API on all questions')},
                  'code_sha256': {p: sha(ROOT / p) for p in ('scripts/evaluate_deep_research.py',
                      'rag/deep_research.py', 'rag/pipeline_v2.py', 'rag/generation.py', 'rag/benchmark.py')}}
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        target = args.resume or ROOT / 'runtime/experiments' / f'deep-research-{stamp}-{uuid4().hex[:8]}'
        previous = read_json(target / 'design.json')
        if args.resume and previous != design:
            raise ValueError('Cannot resume with a different design, snapshot or dataset')
        atomic_json(target / 'design.json', design)
        print(f'Output: {target}', flush=True)
        # Load the local models once before measuring per-question latency.
        if not all((target / (q['question_id'] + '.json')).exists() for q in questions):
            pipeline.retrieve('RAG 论文有哪些常见的评估指标？', auto_route=False)
        rows = []
        for i, question in enumerate(questions, 1):
            qid = question['question_id']
            path = target / f'{qid}.json'
            row = read_json(path)
            if row is None:
                row = run_case(pipeline, question, pages, snapshot, settings, retrieval_only=args.retrieval_only)
                atomic_json(path, row)
            rows.append(row)
            scores = ' -> '.join(f"{row['metrics'][s]['evidence_unit_recall']:.0%}" for s in STAGES[:3]) if row['answerable'] else 'unsupported control'
            print(f"[{i}/{len(questions)}] {qid}: {scores} | {len(row['followups'])} followups | {row['research']['stop_reason']} | "
                  f"{row['research']['usage'].get('total_tokens', 0)} tokens", flush=True)
        positive = [r for r in rows if r['answerable']]
        groups = {'all_answerable': summarize_group(positive),
                  **{k: summarize_group([r for r in positive if r['category'] == k])
                     for k in ('single_paper', 'cross_paper')},
                  **{k: summarize_group([r for r in positive if r['source_scope'] == k])
                     for k in ('named_papers', 'open_corpus_anchor')},
                  **{k: summarize_group([r for r in positive if r['question_id'] in ids])
                     for k, ids in metadata['cohorts'].items()},
                  'unsupported': summarize_group([r for r in rows if not r['answerable']])}
        limitations = [metadata['limitations'], 'Evidence coverage is not answer correctness.',
                       'Product comparison does not isolate follow-up policy against fixed extra retrieval.']
        if not args.retrieval_only:
            limitations.append('Cited context scores full cited passages, not entailment of the cited excerpt.')
        atomic_json(target / 'summary.json', {'design': design, 'groups': groups, 'questions': len(rows),
                                             'limitations': limitations})
        print(f'Summary: {target / "summary.json"}', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
