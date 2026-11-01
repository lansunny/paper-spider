"""Separate QASPER v0.3 paper text from labels; keep official train/dev boundaries."""
import hashlib
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')

def normalize(text):
    return re.sub(r'\s+', ' ', text).strip()

def main():
    archive = DATA / 'qasper-train-dev-v0.3.tgz'
    source_url = 'https://qasper-dataset.s3-us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz'
    raw = DATA / 'qasper_raw'
    raw.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        for name in ['qasper-train-v0.3.json', 'qasper-dev-v0.3.json', 'README.md']:
            (raw / name).write_bytes(bundle.extractfile(name).read())
    statistics = {}
    paper_sets = {}
    for split, starter_count in [('train', 40), ('dev', 20)]:
        papers = json.loads((raw / f'qasper-{split}-v0.3.json').read_text(encoding='utf-8'))
        paper_sets[split] = set(papers)
        starter = set(sorted(papers, key=lambda key: hashlib.sha256(key.encode()).hexdigest())[:starter_count])
        corpus, labels = [], []
        for paper_id, paper in papers.items():
            paragraphs = []
            for section_index, section in enumerate(paper['full_text']):
                for paragraph_index, paragraph in enumerate(section['paragraphs']):
                    paragraphs.append({'paragraph_id': f'{paper_id}:s{section_index}:p{paragraph_index}',
                                       'section': section['section_name'], 'text': paragraph})
            mapping = {}
            for paragraph in paragraphs:
                mapping.setdefault(normalize(paragraph['text']), []).append(paragraph['paragraph_id'])
            corpus.append({'paper_id': paper_id, 'official_split': split, 'title': paper['title'],
                           'abstract': paper['abstract'], 'paragraphs': paragraphs,
                           'figures_and_tables': paper.get('figures_and_tables', []),
                           'source': source_url, 'version': '0.3', 'page_numbers_available': False})
            for question in paper['qas']:
                annotations = []
                for answer in question['answers']:
                    value = answer['answer']
                    groups, unmatched, visual = [], [], []
                    for evidence in value['evidence']:
                        if evidence.startswith('FLOAT SELECTED'):
                            visual.append(evidence)
                        elif normalize(evidence) in mapping:
                            groups.append(mapping[normalize(evidence)])
                        else:
                            unmatched.append(evidence)
                    annotations.append({'annotation_id': answer['annotation_id'], 'answer': value,
                                        'evidence_paragraph_groups': groups, 'unmatched_evidence': unmatched,
                                        'visual_evidence': visual,
                                        'text_retrieval_eligible': bool(groups) and not unmatched and not visual and not value['unanswerable']})
                labels.append({'question_id': question['question_id'], 'paper_id': paper_id,
                               'question': question['question'], 'official_split': split,
                               'language': 'en', 'annotations': annotations})
        write_rows(DATA / 'qasper_corpus' / f'{split}.jsonl', corpus)
        write_rows(DATA / 'evaluation' / f'qasper_{split}.jsonl', labels)
        write_rows(DATA / 'starter' / f'{split}_corpus.jsonl', [row for row in corpus if row['paper_id'] in starter])
        write_rows(DATA / 'evaluation' / f'starter_{split}.jsonl', [row for row in labels if row['paper_id'] in starter])
        statistics[split] = {'papers': len(corpus), 'questions': len(labels),
                             'paragraphs': sum(len(row['paragraphs']) for row in corpus),
                             'starter_papers': len(starter), 'starter_questions': sum(row['paper_id'] in starter for row in labels),
                             'text_retrieval_eligible_questions': sum(any(a['text_retrieval_eligible'] for a in row['annotations']) for row in labels),
                             'has_unanswerable_annotation': sum(any(a['answer']['unanswerable'] for a in row['annotations']) for row in labels),
                             'has_visual_annotation': sum(any(a['visual_evidence'] for a in row['annotations']) for row in labels)}
    assert not paper_sets['train'] & paper_sets['dev'], 'Official paper splits overlap'
    report = {'source': source_url, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'version': '0.3', 'statistics': statistics, 'paper_split_overlap': 0,
              'test_status': 'Not downloaded: official guessed test URL returned 403. Official dev reserved locally for final validation, not called test.',
              'license': 'CC BY 4.0 reported by Hugging Face QASPER loader; retain attribution and source README',
              'license_reference': 'https://github.com/huggingface/datasets/blob/2.3.2/datasets/qasper/qasper.py',
              'caveats': ['QASPER covers diverse NLP, not only RAG.', 'No PDF pages in structured data.',
                          'Figure/table captions only; image pack not downloaded.', 'No questions or answers in corpus files.',
                          'Keep all annotator answers and disagreements; no merged fabricated gold answer.']}
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports/qasper_preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(statistics, indent=2))

if __name__ == '__main__':
    main()
