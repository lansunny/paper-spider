"""Within-paper BM25 evidence retrieval on official QASPER training questions."""
import argparse
import json
import re
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi
ROOT = Path(__file__).resolve().parents[1]

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

def tokens(text):
    return re.findall(r'[a-z0-9]+', text.lower())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'dev'], default='train')
    parser.add_argument('--allow-dev', action='store_true')
    args = parser.parse_args()
    if args.split == 'dev' and not args.allow_dev:
        parser.error('Official dev is reserved for final validation. Pass --allow-dev explicitly.')
    papers = {paper['paper_id']: paper for paper in rows(ROOT / 'data/starter' / f'{args.split}_corpus.jsonl')}
    indexes = {}
    for paper_id, paper in papers.items():
        paragraphs = paper['paragraphs']
        if paragraphs and any(tokens(p['text']) for p in paragraphs):
            indexes[paper_id] = BM25Okapi([tokens(p['text']) for p in paragraphs])
    results = []
    questions = rows(ROOT / 'data/evaluation' / f'starter_{args.split}.jsonl')
    for question in questions:
        gold = [annotation for annotation in question['annotations'] if annotation['text_retrieval_eligible']]
        paper_id = question['paper_id']
        if not gold or paper_id not in indexes:
            continue
        scores = indexes[paper_id].get_scores(tokens(question['question']))
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:5]
        hits = [papers[paper_id]['paragraphs'][i]['paragraph_id'] for i in ranked]
        recalls = [sum(bool(set(group) & set(hits)) for group in a['evidence_paragraph_groups']) / len(a['evidence_paragraph_groups']) for a in gold]
        relevant = {pid for a in gold for group in a['evidence_paragraph_groups'] for pid in group}
        results.append({'question_id': question['question_id'], 'paper_id': paper_id,
                        'evidence_recall_at_5_best_annotation': max(recalls),
                        'mrr_at_5_any_annotation': next((1 / rank for rank, hit in enumerate(hits, 1) if hit in relevant), 0),
                        'retrieved_paragraph_ids': hits})
    report = {'task': 'within-paper evidence retrieval; known paper_id; not open-domain or generation evaluation',
              'official_split': args.split, 'papers': len(papers), 'questions': len(questions),
              'evaluated': len(results), 'excluded': len(questions) - len(results),
              'exclusion_rule': 'No complete matched text-only answerable annotation',
              'metrics': {key: round(sum(row[key] for row in results) / len(results), 4) for key in ['evidence_recall_at_5_best_annotation', 'mrr_at_5_any_annotation']},
              'annotation_policy': 'Best complete text-only annotation recall; MRR accepts evidence from any such annotation.',
              'results': results}
    target = ROOT / 'reports' / f'qasper_bm25_{args.split}.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'results'}, indent=2))

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
