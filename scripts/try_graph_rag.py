"""Inspect opt-in graph-assisted retrieval against V2 without changing either index."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.pipeline_graph import GraphAugmentedPipeline
from rag.pipeline_v2 import load_v2_settings
from rag.paper_graph.schema import LABELS
from rag.storage import atomic_json


def brief(result):
    return {'trace_id': result['trace_id'], 'snapshot_id': result['snapshot_id'],
        'evidence_chunks': len(result['evidence']), 'context_chars': result['context_chars'],
        'rerank_candidates': result['candidate_counts']['rerank_candidates'],
        'evidence_paper_ids': sorted({h['paper_id'] for h in result['evidence']}),
        'elapsed_ms': result['elapsed_ms']}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', default='有没有类似 ARES 的 RAG 自动评估方法？请比较如何衡量回答对检索上下文的忠实性（Faithfulness），以及是否需要人工标注。')
    parser.add_argument('--dimension', choices=['auto', 'topic', 'method', 'experiment', 'conclusion'], default='method')
    parser.add_argument('--mode', choices=['bm25', 'dense', 'hybrid'], default='hybrid')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--compare', action='store_true', help='Also run the unchanged V2 baseline')
    parser.add_argument('--generate', action='store_true', help='Generate one answer for the graph result using the configured API')
    parser.add_argument('--no-rerank', action='store_true')
    args = parser.parse_args()
    pipeline = GraphAugmentedPipeline(load_v2_settings(args.config))
    directory = ROOT / 'runtime/graph_rag' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8])
    directory.mkdir(parents=True)
    try:
        rows = {}
        if args.compare:
            print('Baseline V2:', args.question, flush=True)
            baseline = pipeline.ask(args.question, mode=args.mode, rerank=not args.no_rerank, generate_answer=False)
            atomic_json(directory / 'baseline.json', baseline)
            rows['baseline'] = brief(baseline)
        print('Graph-assisted:', args.question, flush=True)
        result = pipeline.ask(args.question, mode=args.mode, rerank=not args.no_rerank,
            generate_answer=args.generate, graph_retrieval={'enabled': True, 'dimension': args.dimension})
        atomic_json(directory / 'graph.json', result)
        info = result['graph_retrieval']
        rows['graph'] = brief(result)
        print('Graph:', info['status'], '|', LABELS.get(info['dimension'], info['dimension']), '|', info['reason'], flush=True)
        for path in info.get('paths', []):
            print(f"  {path['source_title']} -> {path['target_title']} | cosine={path['weight']:.3f}", flush=True)
        for index, hit in enumerate(result['evidence'], 1):
            print(f"  S{index} [{hit.get('retrieval_origin','baseline')}] {hit['title']} | PDF {hit['pdf_page']}", flush=True)
        if args.compare:
            original = {h['chunk_id'] for h in baseline['evidence']}
            rows['observation'] = {'new_context_chunk_ids': [h['chunk_id'] for h in result['evidence'] if h['chunk_id'] not in original],
                'graph_context_chunk_ids': info.get('context_graph_chunk_ids', []),
                'fixed_context_budget': pipeline.settings.context_char_budget,
                'quality_evaluated': False, 'note': 'Candidate/evidence differences are not accuracy improvements.'}
        if result.get('answer'):
            print('\nAnswer:', result['answer'], flush=True)
        atomic_json(directory / 'summary.json', {'question': args.question, **rows})
        print('Saved:', directory, flush=True)
    finally:
        pipeline.store.close()


if __name__ == '__main__':
    main()
