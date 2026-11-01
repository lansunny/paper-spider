"""One final heldout acceptance of the frozen pipeline, with no generation API.

Default freezes/preflights the baseline. --run evaluates all heldout questions.
A completed acceptance is replayed; interrupted runs resume saved questions.
No --limit, variants, gold paper filters, or parameter tuning are offered.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_questions, load_source_pages, score_evidence
from rag.config import load_settings
from rag.frozen_baseline import freeze_baseline
from rag.generation import select_context
from rag.pipeline import Pipeline
from rag.storage import atomic_json, read_json

STAGES = ['union', 'rrf20', 'after_top5', 'context_budget']


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if not row['answerable']:
            continue
        groups['all_answerable'].append(row)
        groups[row['scope']].append(row)
    return {'questions': len(rows), 'answerable': sum(r['answerable'] for r in rows),
            'unsupported': sum(not r['answerable'] for r in rows),
            'groups': [{'scope': scope, 'questions': len(items), 'stages': {
                stage: {'evidence_coverage': sum(r['metrics'][stage]['evidence_unit_recall'] for r in items)/len(items),
                        'complete_evidence_rate': sum(r['metrics'][stage]['all_required_evidence'] for r in items)/len(items)}
                for stage in STAGES}} for scope, items in sorted(groups.items())],
            'notes': 'Agent-annotated, not independently human-verified. Evidence anchors are not exhaustive; scores are not answer accuracy. Unsupported questions have no positive retrieval gold and require separate generation review.'}


def show_summary(summary, target):
    for group in summary['groups']:
        print(group['scope'], f"{group['questions']} questions", ' | '.join(
            f"{stage}: {v['evidence_coverage']:.1%}/{v['complete_evidence_rate']:.1%}" for stage, v in group['stages'].items()))
    print('Unsupported questions (retrieval scores excluded):', summary['unsupported'])
    print('Acceptance:', target/'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Execute the one final local retrieval acceptance')
    args = parser.parse_args()
    cfg = load_settings()
    frozen = freeze_baseline(cfg, ROOT)
    target = ROOT/'runtime/acceptance'/('heldout-'+frozen['spec_sha256'][:12])
    print('Frozen baseline:', frozen['spec_sha256'][:12], cfg.collection, flush=True)
    state = read_json(target/'run.json')
    if state and state['status'] == 'completed':
        print('Acceptance already completed; displaying saved results.')
        show_summary(read_json(target/'summary.json'), target)
        return
    if not args.run:
        print('Baseline frozen. Add --run for heldout acceptance; no model/API loaded.')
        return
    # Configuration/code are frozen before any heldout labels or scores are read.
    questions = load_questions('heldout')
    digest = hashlib.sha256((ROOT/'data/evaluation/paper_benchmark/heldout.jsonl').read_bytes()).hexdigest()
    if state and (state['heldout_sha256'] != digest or state['spec_sha256'] != frozen['spec_sha256']):
        raise ValueError('Heldout data or baseline changed; do not resume this acceptance')
    state = {'status': 'running', 'spec_sha256': frozen['spec_sha256'], 'heldout_sha256': digest,
             'question_ids': [q['question_id'] for q in questions], 'split': 'heldout',
             'lexical_policy': 'fixed English translations; no automatic rewriting',
             'gold_filter': False, 'generation_api': False}
    atomic_json(target/'run.json', state)
    pipeline = None
    try:
        pipeline = Pipeline(cfg)
        manifest, _ = pipeline.snapshot()
        pipeline.store.validate_collection()
        if manifest['snapshot_id'] != frozen['spec']['snapshot_id'] or pipeline.store.count(manifest['snapshot_id']) != manifest['chunks']:
            raise ValueError('Vector snapshot does not match the frozen baseline')
        # Gold documents only load scoring source text; they never filter retrieval.
        pages = load_source_pages({pid for q in questions for pid in q['gold_paper_ids']})
        if any(manifest['documents'].get(pid) != sha for pid, sha, _ in pages):
            raise ValueError('Annotated PDF differs from indexed document')
        rows = []
        for number, question in enumerate(questions, 1):
            qid = question['question_id']
            saved = read_json(target/f'{qid}.json')
            if saved:
                if saved['question_sha256'] != hashlib.sha256(json.dumps(question, sort_keys=True).encode()).hexdigest():
                    raise ValueError('Saved acceptance row no longer matches this question')
                row = saved
            else:
                result = pipeline.retrieve(question['question'], mode='hybrid', top_k=5, rerank=True,
                                           lexical_query=question['question_en'], collect_trace=True)
                if result['snapshot_id'] != frozen['spec']['snapshot_id']:
                    raise ValueError('Index changed during acceptance')
                context, used = select_context(result['reranked_candidates'], cfg.context_char_budget)
                lists = {'union': result['rankings']['union'], 'rrf20': result['rankings']['rrf20'],
                         'after_top5': result['evidence'], 'context_budget': context}
                metrics = {stage: score_evidence(question, hits, pages, k=max(1, len(hits))) for stage, hits in lists.items()}
                row = {'question_id': qid, 'question': question['question'], 'question_en': question['question_en'],
                       'question_sha256': hashlib.sha256(json.dumps(question, sort_keys=True).encode()).hexdigest(),
                       'answerable': question['answerable'], 'scope': question['source_scope'],
                       'metrics': metrics, 'retrieval': result, 'budget_context': context, 'context_chars': used}
                atomic_json(target/f'{qid}.json', row)
            rows.append({k: row[k] for k in ['question_id', 'answerable', 'scope', 'metrics']})
            values = ' -> '.join(f"{row['metrics'][s]['evidence_unit_recall']:.1%}" for s in STAGES) if row['answerable'] else 'unsupported: generation review required'
            print(f'[{number}/{len(questions)}] {qid}: {values}', flush=True)
        summary = summarize(rows)
        atomic_json(target/'summary.json', summary)
        atomic_json(target/'run.json', {**state, 'status': 'completed'})
        show_summary(summary, target)
    except Exception:
        atomic_json(target/'run.json', {**state, 'status': 'failed_partial_results_retained'})
        raise
    finally:
        if pipeline:
            pipeline.store.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
