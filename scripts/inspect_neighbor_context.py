"""Learning exercise: expand D's saved top-five hits with adjacent source chunks.

This is a two-question context-completeness diagnostic, not a fixed-budget
retrieval benchmark. It reads saved results; no model or database calls occur.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_questions, load_source_pages, score_evidence
from rag.config import load_settings
from rag.ingestion import current_chunks


def expand_neighbors(hits, chunks, radius=1):
    # Match immutable provenance; do not cross papers, document versions or pages.
    lookup = {(c.paper_id, c.document_sha256, c.pdf_page, c.chunk_index): c for c in chunks}
    expanded, seen = [], set()
    for hit in hits:
        for offset in range(-radius, radius + 1):
            key = (hit['paper_id'], hit['document_sha256'], hit['pdf_page'], hit['chunk_index'] + offset)
            chunk = lookup.get(key)
            if chunk is not None and chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                expanded.append(chunk.payload())
    return expanded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runtime/experiments/20261005-200459-42d74384')
    parser.add_argument('--question-ids', nargs='+', default=['paper005', 'paper009'])
    args = parser.parse_args()
    questions = {q['question_id']: q for q in load_questions('dev')}
    manifest, chunks = current_chunks(load_settings())
    run = json.loads((args.run / 'run.json').read_text(encoding='utf-8'))
    if run['snapshot_id'] != manifest['snapshot_id']:
        raise ValueError('Saved experiment and current snapshot differ.')
    for qid in args.question_ids:
        question = questions[qid]
        if not question['answerable']:
            raise ValueError('Select an answerable development question.')
        result = json.loads((args.run / f'{qid}_D.json').read_text(encoding='utf-8'))
        hits = result['rankings']['after_top5']
        expanded = expand_neighbors(hits, chunks)
        pages = load_source_pages(question['gold_paper_ids'])
        print(f"\n{qid}: {question['question']}")
        print('原始前5中的论文页码和分块序号：')
        for hit in hits:
            print(f"  {hit['title']} | PDF {hit['pdf_page']} | chunk {hit['chunk_index']}")
        for label, context in [('原始前5', hits), ('同页前后各扩1块，按chunk ID去重', expanded)]:
            score = score_evidence(question, context, pages, k=len(context))
            chars = sum(len(h['text']) for h in context)
            print(f"{label}: {len(context)}块，正文字符合计{chars}，证据覆盖{score['evidence_unit_recall']:.1%}，齐全={score['all_required_evidence']}")
            for unit in score['units']:
                coverage = [[round(v, 3) for v in group] for group in unit['alternative_span_coverages']]
                print(f"  {unit['unit_id']}: 每段原文覆盖比例={coverage}")
    print('\n这是允许增加上下文长度的诊断。字符合计包含重叠文本，尚未做文本合并、预算截断或生成评估。')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
