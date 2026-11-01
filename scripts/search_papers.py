"""Local lexical search over parsed paper pages. English queries; no LLM or embedding."""
import argparse
import json
import re
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi
ROOT = Path(__file__).resolve().parents[1]

def tokens(text):
    return re.findall(r'[a-z0-9]+', text.lower())

def load_passages():
    passages = []
    for source in sorted((ROOT / 'data/parsed').glob('*.jsonl')):
        for line in source.read_text(encoding='utf-8').splitlines():
            page = json.loads(line)
            # Paragraph/block grouping baseline with a character budget; not the final semantic splitter.
            buffer = ''
            for block in page['blocks']:
                if buffer and len(buffer) + len(block['text']) > 1800:
                    passages.append({'paper_id': page['paper_id'], 'pdf_page': page['pdf_page'], 'text': buffer, 'source_url': page['source_url']})
                    buffer = ''
                buffer += block['text'] + '\n'
            if buffer.strip():
                passages.append({'paper_id': page['paper_id'], 'pdf_page': page['pdf_page'], 'text': buffer, 'source_url': page['source_url']})
    return passages


class PaperRetriever:
    """Existing lexical baseline shared by the CLI and the small web shell."""

    def __init__(self):
        self.passages = load_passages()
        self.index = BM25Okapi([tokens(p['text']) for p in self.passages]) if self.passages else None

    def search(self, query, top_k=5, paper_ids=None):
        if self.index is None:
            raise ValueError('尚无可检索片段，请先运行 PDF 解析。')
        if not tokens(query):
            raise ValueError('此命令检索英文论文文本，请使用英文关键词；中文全文检索请使用工作台或 main.py --task ask。')
        scores = self.index.get_scores(tokens(query))
        allowed = set(paper_ids or [])
        candidates = [i for i, p in enumerate(self.passages) if not allowed or p['paper_id'] in allowed]
        ranked = sorted(candidates, key=lambda i: -float(scores[i]))[:top_k]
        return [{**self.passages[i], 'score': round(float(scores[i]), 4)} for i in ranked if scores[i] > 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query')
    args = parser.parse_args()
    try:
        result = PaperRetriever().search(args.query)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
