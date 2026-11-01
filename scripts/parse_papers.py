"""Page/block extraction baseline. Preserve bounding boxes; no claim of table reconstruction."""
import hashlib
import json
import re
from pathlib import Path
import pymupdf
ROOT = Path(__file__).resolve().parents[1]

def main():
    catalog = json.loads((ROOT / 'data/paper_manifest.json').read_text(encoding='utf-8'))
    output = ROOT / 'data/parsed'
    imports = ROOT / 'data/import/papers'
    previews = ROOT / 'reports/previews'
    for directory in [output, imports, previews]:
        directory.mkdir(parents=True, exist_ok=True)
    report, total_pages = [], 0
    for paper in catalog:
        if paper['status'] != 'downloaded':
            continue
        records, text_pages, sparse, tables = [], [], [], []
        with pymupdf.open(ROOT / paper['pdf_path']) as pdf:
            for index, page in enumerate(pdf):
                blocks = [{'bbox': list(block[:4]), 'text': block[4], 'block_number': block[5]}
                          for block in page.get_text('blocks', sort=False) if block[6] == 0]
                text = '\n\n'.join(block['text'].strip() for block in blocks)
                # Preserve PDF content-stream block order. Store bbox for later layout-aware parsing.
                record = {'paper_id': paper['paper_id'], 'pdf_page': index + 1,
                          'width': page.rect.width, 'height': page.rect.height, 'blocks': blocks,
                          'text': text, 'pdf_sha256': paper['sha256'], 'source_url': paper['landing_url']}
                records.append(record)
                text_pages.append(f'## PDF page {index + 1}\n\n{text}')
                if len(text.strip()) < 150:
                    sparse.append(index + 1)
                if re.search(r'\bTable\s+\d+\s*[:.]', text):
                    tables.append(index + 1)
            title_words = set(re.findall(r'[a-z0-9]+', paper['title'].lower()))
            first_words = set(re.findall(r'[a-z0-9]+', records[0]['text'].lower()))
            title_coverage = len(title_words & first_words) / max(1, len(title_words))
            total_pages += len(records)
            if paper['paper_id'] in ['2024.eacl-demo.16', '2020.emnlp-main.550', '2025.findings-acl.199']:
                for index in sorted({0, min(2, len(pdf)-1)}):
                    pdf[index].get_pixmap(matrix=pymupdf.Matrix(1.1, 1.1)).save(previews / f'{paper["paper_id"]}-p{index+1}.png')
        target = output / f'{paper["paper_id"]}.jsonl'
        target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
        (imports / f'{paper["paper_id"]}.md').write_text(f'# {paper["title"]}\n\nPaper ID: {paper["paper_id"]}\nSource: {paper["landing_url"]}\n\n' + '\n\n'.join(text_pages), encoding='utf-8')
        report.append({'paper_id': paper['paper_id'], 'pages': len(records), 'characters': sum(len(r['text']) for r in records),
                       'title_word_coverage_first_page': round(title_coverage, 3), 'sparse_pages': sparse,
                       'table_caption_candidate_pages': tables, 'parsed_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    summary = {'papers': len(report), 'pages': total_pages, 'parser': f'PyMuPDF {pymupdf.VersionBind}',
               'status': 'baseline extraction; sampled visual QA only; tables/formulas/reading order require advanced parser evaluation',
               'title_check_failures': [r['paper_id'] for r in report if r['title_word_coverage_first_page'] < 0.7], 'details': report}
    (ROOT / 'reports/pdf_parse_quality.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'details'}, indent=2))

if __name__ == '__main__':
    main()
