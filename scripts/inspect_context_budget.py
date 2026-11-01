"""Compare source-span merging and a common serialized-evidence budget offline.

Default: two diagnostic questions from the saved D run, budget=8000 characters.
Only loads source text and saved retrieval results; never invokes a model.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_questions, load_source_pages, score_evidence
from rag.config import load_settings
from rag.context_windows import merge_source_windows, bounded_neighbor_windows
from rag.generation import select_context
from rag.ingestion import current_chunks
from rag.storage import atomic_json
from scripts.inspect_neighbor_context import expand_neighbors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runtime/experiments/20261005-200459-42d74384')
    parser.add_argument('--question-ids', nargs='+', default=['paper005', 'paper009'])
    parser.add_argument('--budget', type=int, default=8000)
    parser.add_argument('--max-seed-chars', type=int, default=2400,
                        help='Body character cap per seed expansion, before overlap merging')
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error('--budget must be positive')
    if args.max_seed_chars <= 0:
        parser.error('--max-seed-chars must be positive')
    manifest, chunks = current_chunks(load_settings())
    run = json.loads((args.run / 'run.json').read_text(encoding='utf-8'))
    if run['snapshot_id'] != manifest['snapshot_id']:
        raise ValueError('Saved experiment and current index snapshots differ.')
    questions = {q['question_id']: q for q in load_questions('dev')}
    results = []
    for qid in args.question_ids:
        question = questions[qid]
        if not question['answerable']:
            raise ValueError('Choose an answerable development question.')
        result = json.loads((args.run / f'{qid}_D.json').read_text(encoding='utf-8'))
        seeds = result['rankings']['after_top5']
        expanded = expand_neighbors(seeds, chunks)
        # Source loading includes every retrieved paper. Gold IDs are only needed
        # by scoring, never by neighbor selection, merging or budget allocation.
        source_ids = {h['paper_id'] for h in expanded} | set(question['gold_paper_ids'])
        pages = load_source_pages(source_ids)
        original_windows = merge_source_windows(seeds, pages)
        expanded_windows = merge_source_windows(expanded, pages)
        baseline, baseline_size = select_context(original_windows, args.budget)
        bounded, bounded_size = select_context(expanded_windows, args.budget)
        limited_windows = merge_source_windows(
            bounded_neighbor_windows(seeds, chunks, pages, args.max_seed_chars), pages)
        limited, limited_size = select_context(limited_windows, args.budget)
        stages = [('原始前5', seeds, None), ('扩展后按ID去重', expanded, None),
                  ('扩展后合并重叠原文', expanded_windows, None),
                  ('原始前5合并＋预算', baseline, baseline_size),
                  ('相邻扩展合并＋预算', bounded, bounded_size),
                  ('限长相邻扩展合并＋预算', limited, limited_size)]
        print(f"\n{qid}: {question['question']}")
        print(f'预算方案使用同一个上限：{args.budget} 个序列化证据字符；限长方案每个种子扩展最多{args.max_seed_chars}个正文字符（合并前）。')
        rows = []
        for label, context, serialized_chars in stages:
            score = score_evidence(question, context, pages, k=max(1, len(context)))
            raw_chars = sum(len(h['text']) for h in context)
            print(f"{label}: {len(context)}项，正文{raw_chars}字符，证据覆盖{score['evidence_unit_recall']:.1%}，齐全={score['all_required_evidence']}")
            if serialized_chars is not None:
                print(f'  序列化证据字符用量：{serialized_chars}/{args.budget}')
                print('  保留位置：', [(h['paper_id'], h['pdf_page'], h['char_start'], h['char_end']) for h in context])
            rows.append({'stage': label, 'body_chars': raw_chars, 'serialized_chars': serialized_chars,
                         'metrics': score, 'context': context})
        selected_ids = {h['window_id'] for h in bounded}
        skipped = [{k: h[k] for k in ['window_id', 'paper_id', 'pdf_page', 'char_start', 'char_end']}
                   for h in expanded_windows if h['window_id'] not in selected_ids]
        print('  扩展方案因预算跳过的窗口：', [(h['paper_id'], h['pdf_page']) for h in skipped])
        limited_ids = {h['window_id'] for h in limited}
        limited_skipped = [{k: h[k] for k in ['window_id', 'paper_id', 'pdf_page', 'char_start', 'char_end']}
                           for h in limited_windows if h['window_id'] not in limited_ids]
        print('  限长方案因预算跳过的窗口：', [(h['paper_id'], h['pdf_page']) for h in limited_skipped])
        results.append({'question_id': qid, 'stages': rows, 'budget_skipped_windows': skipped,
                        'limited_budget_skipped_windows': limited_skipped})
    target = ROOT / 'runtime/experiments' / ('context-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8])
    atomic_json(target / 'results.json', {'source_run': str(args.run), 'snapshot_id': manifest['snapshot_id'],
                'budget': args.budget, 'max_seed_chars': args.max_seed_chars,
                'per_seed_limit_note': 'Applies before merging; merged windows may be longer. Original seed is always preserved. Extensions use punctuation/paragraph boundary heuristics.',
                'budget_unit': 'sum of serialized individual evidence objects, including citation IDs/title/page; excludes prompt wrappers and answer tokens',
                'selection': 'earliest seed priority; skip whole windows that do not fit; no gold-based selection',
                'scope': 'two-case diagnostic by default, not a heldout or complete generation evaluation',
                'results': results})
    print(f'\n结果：{target / "results.json"}')
    print('字符预算不等于模型token预算。保留种子块；限长扩展按标点/段落边界收缩；总预算超限跳过整段窗口。不按标准答案挑选内容。')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
