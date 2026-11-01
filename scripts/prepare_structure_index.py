"""PyCharm entry: validate/export structural chunks; --build embeds a new index.

Default is a full-corpus offline check using ONLY the local tokenizer, not model
weights. The original retrieval configuration and collection stay unchanged.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import chunk_intervals
from rag.chunking import normalized_blocks
from rag.config import load_settings
from rag.ingestion import source_documents, pdf_pages, document_chunks
from rag.storage import atomic_json, atomic_text
from rag.structural_chunking import ordered_blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/retrieval_structure.json')
    parser.add_argument('--build', action='store_true', help='Load local models and embed into the independent collection')
    args = parser.parse_args()
    cfg = load_settings(args.config)
    if cfg.chunk_strategy != 'layout-blocks':
        raise ValueError('Choose a layout-blocks configuration')
    if args.build:
        from rag.embeddings import LocalModels
        from rag.ingestion import build_index
        from rag.vector_store import QdrantStore
        store = QdrantStore(cfg)
        try:
            result = build_index(cfg, LocalModels(cfg), store, progress=lambda **s: print(json.dumps(s), flush=True))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            store.close()
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / cfg.embedding_path), local_files_only=True, use_fast=True)
    # Avoid model-context warnings when measuring uncut source blocks.
    tokenizer.model_max_length = 10**9
    documents = source_documents(cfg)
    total, pages_count, tokens, methods = [], 0, [], Counter()
    for number, document in enumerate(documents, 1):
        pages = pdf_pages(document)
        source_pages, labels = {}, {}
        for page in pages:
            key = (document['paper_id'], document['sha256'], page['pdf_page'])
            source_pages[key] = normalized_blocks(page['blocks'])[0]
            labels[page['pdf_page']] = {r['number']: (r['column'], r['region']) for r in ordered_blocks(page)[1]}
        chunks = list(document_chunks(document, tokenizer, cfg))
        intervals = chunk_intervals(chunks, source_pages)
        for key, text in source_pages.items():
            cursor = 0
            for start, end in intervals.get(key, []):
                if text[cursor:start].strip():
                    raise ValueError(f'Lost source text before {start}: {key}')
                cursor = max(cursor, end)
            if text[cursor:].strip():
                raise ValueError(f'Lost source text after {cursor}: {key}')
        for chunk in chunks:
            n = len(tokenizer(chunk.text, add_special_tokens=False, truncation=False)['input_ids'])
            full = len(tokenizer(chunk.text, add_special_tokens=True, truncation=False)['input_ids'])
            if n > cfg.chunk_size or full > cfg.embedding_max_length:
                raise ValueError(f'Token overflow in {chunk.chunk_id}')
            memberships = {labels[chunk.pdf_page][s.block_number] for s in chunk.source_spans}
            if memberships != {(chunk.layout_column, chunk.layout_region)}:
                raise ValueError('Chunk crosses assigned layout columns/regions')
            tokens.append(n)
            methods[chunk.split_method] += 1
        total.extend(chunks)
        pages_count += len(pages)
        if number % 25 == 0 or number == len(documents):
            print(f'已检查 {number}/{len(documents)} 篇，累计 {len(total)} 块', flush=True)
    tokens.sort()
    output = ROOT / 'runtime/structure_preview'
    atomic_text(output / 'chunks.jsonl', ''.join(c.model_dump_json() + '\n' for c in total))
    report = {'status': 'passed', 'collection_for_next_build': cfg.collection,
              'index_profile': cfg.index_profile(), 'settings': cfg.model_dump(),
              'documents': len(documents), 'pages': pages_count, 'chunks': len(total),
              'body_tokens': {'p50': tokens[len(tokens)//2], 'p90': tokens[int(len(tokens)*.9)],
                              'max': max(tokens)}, 'split_methods': dict(methods),
              'all_nonwhitespace_source_text_preserved': True,
              'all_source_spans_exact': True, 'assigned_column_region_crossings': 0,
              'token_overflows': 0, 'vectors_built': False,
              'documents_sha256': {d['paper_id']: d['sha256'] for d in documents},
              'chunks_file_sha256': hashlib.sha256((output / 'chunks.jsonl').read_bytes()).hexdigest(),
              'limitation': 'Geometry/heading/sentence heuristics; these checks prove provenance and bounds, not perfect semantic reading order or better retrieval.'}
    atomic_json(output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in {'documents_sha256', 'settings'}}, ensure_ascii=False, indent=2))
    print(f'预览：{output}; 正式建库请使用 --build')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
