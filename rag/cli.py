"""Shared experiment entry points used by main.py and PyCharm run configurations."""
import argparse
import json
import sys
from pathlib import Path

from rag.config import RetrievalSettings, load_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Independent retrieval config JSON')
    parser.add_argument('task', choices=['models', 'index', 'ask', 'evaluate', 'doctor'])
    parser.add_argument('--corpus', choices=['papers', 'qasper-train', 'qasper-dev'], default='papers')
    parser.add_argument('--collection')
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--endpoint', help='Optional Hugging Face-compatible download mirror URL')
    parser.add_argument('--query', default='RAGAs 如何评价回答是否忠实于检索证据？')
    parser.add_argument('--lexical-query', help='Optional English query for BM25; dense/rerank use the original question')
    parser.add_argument('--mode', choices=['bm25', 'dense', 'hybrid'], default='hybrid')
    parser.add_argument('--no-rerank', action='store_true')
    parser.add_argument('--no-generate', action='store_true')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--variants', nargs='+', default=['bm25', 'dense', 'hybrid', 'hybrid-rerank'])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--allow-dev', action='store_true')
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    values = load_settings(args.config).model_dump()
    values['corpus'] = args.corpus
    if args.collection:
        values['collection'] = args.collection
    elif args.corpus != 'papers':
        values['collection'] += '_' + args.corpus.replace('-', '_')
    if args.chunk_size is not None or args.overlap is not None:
        if not args.collection:
            parser.error('For a chunking experiment, also supply an independent --collection.')
    for attribute, field in [('chunk_size', 'chunk_size'), ('overlap', 'chunk_overlap'), ('device', 'device')]:
        if getattr(args, attribute) is not None:
            values[field] = getattr(args, attribute)
    settings = RetrievalSettings(**values)
    if args.corpus == 'qasper-dev' and args.task in {'index', 'evaluate'} and not args.allow_dev:
        parser.error('dev is reserved for final validation; pass --allow-dev explicitly.')
    if args.task == 'models':
        from rag.downloads import prepare_models
        prepare_models(settings, endpoint=args.endpoint, progress=lambda s: print(s, flush=True))
        return 0
    from rag.pipeline import Pipeline
    pipeline = Pipeline(settings)
    try:
        if args.task == 'index':
            from rag.ingestion import build_index
            result = build_index(settings, pipeline.models, pipeline.store,
                                 progress=lambda **s: print(json.dumps(s), flush=True), force=args.force)
        elif args.task == 'ask':
            result = pipeline.ask(args.query, mode=args.mode, top_k=args.top_k,
                                  rerank=not args.no_rerank, generate_answer=not args.no_generate,
                                  lexical_query=args.lexical_query)
        elif args.task == 'evaluate':
            from rag.evaluation import evaluate
            path, summary = evaluate(pipeline, args.variants, args.top_k, args.limit, args.allow_dev,
                                     progress=lambda s: print(s, flush=True))
            result = {'report': str(path), 'summaries': summary}
        else:
            from rag.generation import generation_settings
            from rag.ingestion import current_manifest
            import torch
            pipeline.store.ensure_collection()
            manifest = current_manifest(settings)
            result = {'python': sys.executable, 'cuda': torch.cuda.is_available(),
                      'qdrant': pipeline.store.client.info().version,
                      'collection': settings.collection,
                      'indexed_chunks': manifest['chunks'] if manifest else 0,
                      'generation_key_configured': generation_settings().configured}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        pipeline.store.close()


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
