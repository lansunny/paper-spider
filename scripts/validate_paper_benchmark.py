"""Validate custom source spans and all existing QASPER label references offline."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_questions, load_source_pages
from rag.config import load_settings
from rag.ingestion import current_chunks
from rag.storage import atomic_json


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def validate_qasper():
    """Official labels remain untouched; visual/unmatched evidence is reported."""
    report = {}
    for split in ['train', 'dev']:
        corpus = {r['paper_id']: r for r in read_rows(ROOT / f'data/qasper_corpus/{split}.jsonl')}
        labels = read_rows(ROOT / f'data/evaluation/qasper_{split}.jsonl')
        raw = json.loads((ROOT / f'data/qasper_raw/qasper-{split}-v0.3.json').read_text(encoding='utf-8'))
        counts = Counter()
        ids = set()
        for row in labels:
            assert row['question_id'] not in ids
            ids.add(row['question_id'])
            paper = corpus[row['paper_id']]
            paragraphs = {p['paragraph_id']: p['text'] for p in paper['paragraphs']}
            official = next(q for q in raw[row['paper_id']]['qas'] if q['question_id'] == row['question_id'])
            assert row['question'] == official['question']
            assert len(row['annotations']) == len(official['answers'])
            for ann, orig in zip(row['annotations'], official['answers'], strict=True):
                assert ann['answer'] == orig['answer'] and ann['annotation_id'] == orig['annotation_id']
                for group in ann['evidence_paragraph_groups']:
                    assert group
                    for paragraph in group:
                        assert paragraph in paragraphs
                        normalized = ' '.join(paragraphs[paragraph].split())
                        assert any(' '.join(e.split()) == normalized for e in ann['answer']['evidence'])
                        counts['validated_paragraph_references'] += 1
                expected = bool(ann['evidence_paragraph_groups']) and not ann['unmatched_evidence'] and not ann['visual_evidence'] and not ann['answer']['unanswerable']
                assert ann['text_retrieval_eligible'] == expected
                counts['annotations'] += 1
                counts['visual_annotations'] += bool(ann['visual_evidence'])
                counts['unmatched_annotations'] += bool(ann['unmatched_evidence'])
                counts['text_eligible_annotations'] += expected
            counts['questions'] += 1
        assert len(labels) == sum(len(p['qas']) for p in raw.values())
        report[split] = {'papers': len(corpus), **dict(counts), 'official_answers_preserved': True}
    train_ids = {r['paper_id'] for r in read_rows(ROOT / 'data/qasper_corpus/train.jsonl')}
    dev_ids = {r['paper_id'] for r in read_rows(ROOT / 'data/qasper_corpus/dev.jsonl')}
    assert not train_ids & dev_ids
    return report


def main():
    rows = load_questions('dev') + load_questions('heldout')
    pids = {pid for row in rows for pid in row['gold_paper_ids']}
    pages = load_source_pages(pids)
    ids, split_papers, unique_spans = set(), defaultdict(set), set()
    span_count = 0
    for row in rows:
        assert row['question_id'] not in ids
        ids.add(row['question_id'])
        split_papers[row['split']].update(row['gold_paper_ids'])
        assert row['question'].strip() and row['question_en'].strip() and row['reference_answer'].strip()
        units = {u['unit_id']: u for u in row['evidence_units']}
        assert len(units) == len(row['evidence_units'])
        assert bool(units) == row['answerable']
        assert all(point['text'] and set(point['evidence_unit_ids']) <= units.keys() for point in row['required_answer_points'])
        if row['answerable']:
            assert all(point['evidence_unit_ids'] for point in row['required_answer_points'])
            evidence_pids = set()
            for unit in units.values():
                assert unit['alternatives'] and all(unit['alternatives'])
                for alternative in unit['alternatives']:
                    for span in alternative:
                        key = (span['paper_id'], span['document_sha256'], span['pdf_page'])
                        page = pages[key]
                        left, right = span['char_start'], span['char_end']
                        assert 0 <= left < right <= len(page)
                        assert page[left:right] == span['quote']
                        assert hashlib.sha256(span['quote'].encode()).hexdigest() == span['quote_sha256']
                        assert span['source_url'].startswith('https://')
                        unique_spans.add((*key, left, right))
                        evidence_pids.add(span['paper_id'])
                        span_count += 1
            assert evidence_pids == set(row['gold_paper_ids'])
        else:
            check = row['absence_check']
            scoped = [t for key, t in pages.items() if key[0] == check['paper_id'] and key[1] == check['document_sha256']]
            assert len(scoped) == check['pages_checked']
            assert check['absent_term'].casefold() not in '\n'.join(scoped).casefold()
    assert not split_papers['dev'] & split_papers['heldout']
    seeds = json.loads((ROOT / 'data/evaluation/cross_paper_seed.json').read_text(encoding='utf-8'))
    assert all(q['annotation_question_id'] in ids for q in seeds['questions'])
    manifest, chunks = current_chunks(load_settings())
    assert pids <= {c.paper_id for c in chunks}
    # Optional snapshot-specific lookup: intersection is not proof of full support.
    by_page = defaultdict(list)
    for c in chunks:
        key = (c.paper_id, c.document_sha256, c.pdf_page)
        if key in pages:
            start = pages[key].find(c.text)
            assert start >= 0 and pages[key].find(c.text, start + 1) < 0
            by_page[key].append((start, start + len(c.text), c.chunk_id))
    mapping = []
    for pid, digest, page, start, end in sorted(unique_spans):
        linked = [cid for left, right, cid in by_page[(pid, digest, page)] if right > start and left < end]
        assert linked
        mapping.append({'paper_id': pid, 'document_sha256': digest, 'pdf_page': page,
                        'char_start': start, 'char_end': end, 'intersecting_chunk_ids': linked})
    target = ROOT / 'data/evaluation/paper_benchmark'
    atomic_json(target / 'current_chunk_mapping.json', {'snapshot_id': manifest['snapshot_id'],
        'meaning': 'Auxiliary span intersections only; not permanent gold, not all intersecting chunks are individually sufficient.',
        'spans': mapping})
    report = {'status': 'passed', 'questions': len(rows), 'source_papers': len(pids),
              'counts_by_split': dict(Counter(r['split'] for r in rows)),
              'evidence_span_references_checked': span_count, 'unique_source_spans': len(unique_spans),
              'gold_paper_split_overlap': 0, 'current_snapshot': manifest['snapshot_id'],
              'current_corpus_documents': len(manifest['documents']), 'current_chunks': len(chunks),
              'all_source_quotes_and_document_hashes_valid': True, 'qasper': validate_qasper(),
              'semantic_validation': 'Source-grounded agent review; mechanical validation does not substitute for independent human semantic review.'}
    atomic_json(target / 'validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
