"""Reproducible within-paper QASPER retrieval ablations; labels never enter indexing."""
import json
import platform
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from rag.config import ROOT
from rag.ingestion import current_manifest
from rag.retrieval import retrieval_metrics
from rag.storage import atomic_json


def evaluate(pipeline, variants, top_k=5, limit=0, allow_dev=False, progress=print):
    cfg = pipeline.settings
    if cfg.corpus not in {'qasper-train', 'qasper-dev'}:
        raise ValueError('Evaluation requires an independently indexed QASPER corpus.')
    split = cfg.corpus.removeprefix('qasper-')
    if split == 'dev' and not allow_dev:
        raise ValueError('dev is reserved for final validation; explicitly pass --allow-dev.')
    path = pipeline.root / 'data/evaluation' / f'starter_{split}.jsonl'
    questions = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    eligible = [q for q in questions if any(a.get('text_retrieval_eligible') for a in q['annotations'])]
    selected = eligible[:limit] if limit else eligible
    if not selected:
        raise ValueError('No eligible evaluation questions.')
    manifest = current_manifest(cfg, pipeline.root)
    if manifest is None:
        raise ValueError('Build the QASPER index before evaluation.')
    results, summaries = [], {}
    modes = {'bm25': ('bm25', False), 'dense': ('dense', False),
             'hybrid': ('hybrid', False), 'hybrid-rerank': ('hybrid', True)}
    for variant in variants:
        if variant not in modes:
            raise ValueError('Unknown experiment variant: ' + variant)
        mode, rerank = modes[variant]
        # Separate cold-start time from per-query latency.
        cold = perf_counter()
        first = selected[0]
        pipeline.retrieve(first['question'], mode, top_k, rerank, [first['paper_id']])
        warmup_ms = (perf_counter() - cold) * 1000
        rows = []
        for index, question in enumerate(selected, 1):
            if current_manifest(cfg, pipeline.root)['snapshot_id'] != manifest['snapshot_id']:
                raise RuntimeError('Index changed during evaluation; restart for a comparable run.')
            started = perf_counter()
            output = pipeline.retrieve(question['question'], mode, top_k, rerank, [question['paper_id']])
            metrics = retrieval_metrics(output['evidence'], question['annotations'])
            rows.append({'variant': variant, 'question_id': question['question_id'],
                         'paper_id': question['paper_id'], **metrics,
                         'latency_ms': (perf_counter() - started) * 1000,
                         'chunk_ids': [hit['chunk_id'] for hit in output['evidence']],
                         'paragraph_ids': [hit['paragraph_ids'] for hit in output['evidence']]})
            if index % 10 == 0 or index == len(selected):
                progress(f'{variant}: {index}/{len(selected)}')
        results.extend(rows)
        summaries[variant] = {
            f'evidence_recall_at_{top_k}_best_annotation': float(np.mean([r['evidence_recall'] for r in rows])),
            f'mrr_at_{top_k}_any_annotation': float(np.mean([r['mrr'] for r in rows])),
            'latency_p50_ms': float(np.percentile([r['latency_ms'] for r in rows], 50)),
            'latency_p95_ms': float(np.percentile([r['latency_ms'] for r in rows], 95)),
            'warmup_ms': warmup_ms,
        }
    report = {'task': 'within-paper text evidence retrieval; known paper_id; no generation evaluation',
              'split': split, 'top_k': top_k, 'available_questions': len(questions),
              'eligible_questions': len(eligible), 'evaluated_questions': len(selected),
              'excluded_ineligible': len(questions) - len(eligible), 'limited_run': bool(limit),
              'annotation_policy': 'Best complete text-only annotation recall; MRR accepts any eligible annotation.',
              'snapshot_id': manifest['snapshot_id'], 'settings': cfg.model_dump(),
              'python': platform.python_version(), 'summaries': summaries, 'results': results,
              'created_at': datetime.now(timezone.utc).isoformat()}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = pipeline.root / 'reports/experiments' / f'{split}-{stamp}.json'
    atomic_json(target, report)
    return target, summaries
