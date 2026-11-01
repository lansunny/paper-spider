"""Evaluate the frozen baseline on test50, retaining old/fresh cohort scores.

Default validates and previews. --run reuses the original 23 acceptance rows and
retrieves only the new 27. No parameter variants, generation API, or tuning.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_questions, load_source_pages, score_evidence
from rag.config import load_settings
from rag.frozen_baseline import freeze_baseline
from rag.generation import select_context
from rag.pipeline import Pipeline
from rag.storage import atomic_json, read_json
from scripts.accept_heldout import summarize, STAGES
from scripts.prepare_test50 import validate_rows


def question_hash(q):
    return hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest()


def validate_cohorts(questions, metadata):
    ids = [q['question_id'] for q in questions]
    old = metadata['cohorts']['previously_evaluated23']
    fresh = metadata['cohorts']['new_questions27']
    if len(ids) != 50 or len(set(ids)) != 50 or len(old) != 23 or len(fresh) != 27:
        raise ValueError('Expected 50 unique questions with 23 old and 27 new IDs')
    if set(old) & set(fresh) or set(old) | set(fresh) != set(ids):
        raise ValueError('Cohorts overlap or do not partition the dataset')
    if Counter(q['category'] for q in questions) != Counter(metadata['categories']):
        raise ValueError('Category counts differ from dataset manifest')
    if sum(q['answerable'] for q in questions) != metadata['answerable']:
        raise ValueError('Answerable count differs from manifest')


def cohort_summary(rows, metadata):
    return {'all50': summarize(rows), **{
        name: summarize([row for row in rows if row['question_id'] in set(ids)])
        for name, ids in metadata['cohorts'].items()}}


def show(summary, target):
    for name, report in summary['cohorts'].items():
        group = next(g for g in report['groups'] if g['scope'] == 'all_answerable')
        print(name, f"total={report['questions']}, answerable={report['answerable']}, unsupported={report['unsupported']}",
              ' | '.join(f"{stage}: {v['evidence_coverage']:.1%}/{v['complete_evidence_rate']:.1%}" for stage, v in group['stages'].items()))
    print('Result:', target/'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    cfg = load_settings()
    frozen = freeze_baseline(cfg, ROOT)
    data_dir = ROOT/'data/evaluation/paper_benchmark'
    path = data_dir/'test50.jsonl'
    if not path.exists():
        raise ValueError('先运行 scripts/prepare_test50.py 准备并验证50题数据。')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = read_json(data_dir/'test50_manifest.json')
    if digest != metadata['dataset_sha256']:
        raise ValueError('test50数据与清单哈希不匹配。')
    questions = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    validate_cohorts(questions, metadata)
    target = ROOT/'runtime/acceptance'/('test50-'+frozen['spec_sha256'][:12]+'-'+digest[:8])
    previous = read_json(target/'run.json')
    if previous and previous['status'] == 'completed':
        print('Displaying completed 50-question acceptance; no new retrieval.')
        show(read_json(target/'summary.json'), target)
        return
    pages = load_source_pages({pid for q in questions for pid in q['gold_paper_ids']})
    validate_rows(questions, pages, load_questions('dev'))
    print('Validated 50 questions: 46 answerable, 4 unsupported; baseline', frozen['spec_sha256'][:12], flush=True)
    if not args.run:
        print('Add --run to evaluate only the new 27 and reuse original 23 results.')
        return
    old_target = ROOT/'runtime/acceptance'/('heldout-'+frozen['spec_sha256'][:12])
    old_state = read_json(old_target/'run.json')
    if not old_state or old_state['status'] != 'completed' or old_state['spec_sha256'] != frozen['spec_sha256']:
        raise ValueError('找不到相同冻结方案的原23题验收，先完成原验收。')
    if old_state['heldout_sha256'] != metadata['original_heldout_sha256']:
        raise ValueError('原23题来源版本不同，不能直接复用。')
    state = {'status': 'running', 'spec_sha256': frozen['spec_sha256'], 'dataset_sha256': digest,
             'question_ids': [q['question_id'] for q in questions], 'cohorts': metadata['cohorts'],
             'settings': cfg.model_dump(), 'generation_api': False, 'retrieval_parameters_changed': False,
             'annotation_human_verified': False,
             'notes': '50 total is an expanded detection set, not 50 untouched heldout samples. Existing 23 are reused; new 27 reported separately.'}
    atomic_json(target/'run.json', state)
    pipeline, rows = None, []
    old_ids = set(metadata['cohorts']['previously_evaluated23'])
    try:
        for number, q in enumerate(questions, 1):
            qid = q['question_id']
            row = read_json(target/f'{qid}.json')
            if row:
                if row['question_sha256'] != question_hash(q):
                    raise ValueError('Saved result no longer matches the question')
                action = 'saved'
            elif qid in old_ids:
                row = read_json(old_target/f'{qid}.json')
                if not row or row['question_sha256'] != question_hash(q):
                    raise ValueError('Original cached acceptance question differs')
                row = {**row, 'cohort': 'previously_evaluated23', 'reused_original_result': True}
                atomic_json(target/f'{qid}.json', row)
                action = 'reused'
            else:
                if pipeline is None:
                    pipeline = Pipeline(cfg)
                    manifest, _ = pipeline.snapshot()
                    pipeline.store.validate_collection()
                    if manifest['snapshot_id'] != frozen['spec']['snapshot_id'] or pipeline.store.count(manifest['snapshot_id']) != manifest['chunks']:
                        raise ValueError('Vector snapshot differs from frozen baseline')
                    if any(manifest['documents'].get(pid) != sha for pid, sha, _ in pages):
                        raise ValueError('Annotation PDF does not match indexed document')
                result = pipeline.retrieve(q['question'], mode='hybrid', top_k=5, rerank=True,
                                           lexical_query=q['question_en'], collect_trace=True)
                if result['snapshot_id'] != frozen['spec']['snapshot_id']:
                    raise ValueError('Index changed during evaluation')
                context, used = select_context(result['reranked_candidates'], cfg.context_char_budget)
                candidates = {'union': result['rankings']['union'], 'rrf20': result['rankings']['rrf20'],
                              'after_top5': result['evidence'], 'context_budget': context}
                metrics = {stage: score_evidence(q, hits, pages, k=max(1, len(hits))) for stage, hits in candidates.items()}
                row = {'question_id': qid, 'question': q['question'], 'question_en': q['question_en'],
                       'question_sha256': question_hash(q), 'answerable': q['answerable'], 'scope': q['source_scope'],
                       'metrics': metrics, 'retrieval': result, 'budget_context': context, 'context_chars': used,
                       'cohort': 'new_questions27', 'reused_original_result': False}
                atomic_json(target/f'{qid}.json', row)
                action = 'new'
            rows.append({k: row[k] for k in ['question_id', 'answerable', 'scope', 'metrics']})
            values = ' -> '.join(f"{row['metrics'][s]['evidence_unit_recall']:.1%}" for s in STAGES) if q['answerable'] else 'unsupported: requires generation review'
            print(f'[{number}/50] {qid} ({action}): {values}', flush=True)
        summary = {'cohorts': cohort_summary(rows, metadata), 'dataset_sha256': digest,
                   'source_papers': metadata['source_papers'], 'new_source_papers': metadata['new_source_papers'],
                   'limitations': metadata['limitations']}
        atomic_json(target/'summary.json', summary)
        atomic_json(target/'run.json', {**state, 'status': 'completed'})
        show(summary, target)
    except Exception:
        atomic_json(target/'run.json', {**state, 'status': 'failed_partial_results_retained'})
        raise
    finally:
        if pipeline:
            pipeline.store.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
