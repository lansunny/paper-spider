"""Prepare or run a resumable four-view paper graph pilot; preview is the default."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import portalocker

from rag.config import load_settings
from rag.embeddings import LocalModels
from rag.generation import generation_settings
from rag.ingestion import current_manifest, paper_catalog, fingerprint
from rag.storage import atomic_json
from rag.paper_graph.build import GraphSettings, prepare, make_profile, read_profile, build_bundle
from rag.paper_graph.schema import DIMENSIONS
from rag.paper_graph.store import PaperGraphStore


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/paper_graph.json')
    parser.add_argument('--retrieval-config', type=Path, default=ROOT / 'configs/retrieval_structure.json')
    parser.add_argument('--limit', type=int, help='Pilot size; defaults to the graph config (50)')
    parser.add_argument('--run', action='store_true', help='Run real extraction/encoding; reuses completed caches')
    parser.add_argument('--stage', choices=['all', 'extract', 'graph'], default='all')
    parser.add_argument('--inspect', metavar='PAPER_ID', help='Inspect published descriptions, sources and neighbors')
    parser.add_argument('--dimension', choices=DIMENSIONS, default='topic')
    args = parser.parse_args()
    if args.inspect:
        store = PaperGraphStore(ROOT)
        paper = store.paper(args.inspect)
        dimension = paper['dimensions'][args.dimension]
        print(paper['title'], args.dimension, dimension['status'])
        print(dimension['description'] or dimension['reason'])
        for claim in dimension['claims']:
            print(claim['field'] + ':', claim['text'])
            for source in claim['sources']:
                print(' ', source['source_id'], 'PDF', source['pdf_page'],
                      f"chars [{source['source_start']},{source['source_end']})", source['study_url'])
        print('Neighbors:', json.dumps(store.neighbors(args.inspect, args.dimension), ensure_ascii=False))
        return 0
    values = json.loads(args.config.read_text(encoding='utf-8'))
    if args.limit is not None:
        values['limit'] = args.limit
    settings = GraphSettings.model_validate(values)
    encoder_settings = load_settings(args.retrieval_config)
    generation = generation_settings().model_copy(update={'max_tokens': settings.max_output_tokens,
                                                          'timeout_seconds': 180})
    snapshot = current_manifest(encoder_settings)
    if not snapshot or snapshot['profile'] != encoder_settings.index_profile():
        raise ValueError('Choose a compatible built paper index')
    papers = [p for p in paper_catalog(ROOT) if snapshot['documents'].get(p['paper_id']) == p['sha256']]
    records = prepare(ROOT, papers, settings, generation)
    if not records or len(records) < settings.limit:
        raise ValueError('Not enough matching papers for this pilot size')
    plan = {'status': 'prepared', 'papers': len(records), 'source_snapshot_id': snapshot['snapshot_id'],
        'model': generation.model, 'prompt_version': 'paper-four-views-v1', 'settings': settings.model_dump(),
        'documents': [{'paper_id': r['paper']['paper_id'], 'title': r['paper']['title'],
            'document_sha256': r['paper']['sha256'], 'source_blocks': len(r['sources']),
            'source_batches': len(r['batches']), 'cache_key': r['cache_key'],
            'cached': read_profile(ROOT, r) is not None} for r in records]}
    work = ROOT / 'runtime/paper_graph/work' / fingerprint([r['cache_key'] for r in records])[:16]
    atomic_json(work / 'plan.json', plan)
    print('Pilot:', len(records), 'papers;', sum(len(r['batches']) for r in records),
          'source batches;', sum(d['cached'] for d in plan['documents']), 'cached papers', flush=True)
    print('Plan:', work / 'plan.json', flush=True)
    if not args.run:
        print('Preview only. Add --run to execute the selected stage.', flush=True)
        return 0
    root = ROOT / 'runtime/paper_graph'
    root.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(root / 'build.lock'), timeout=0):
        if args.stage in {'all', 'extract'}:
            if not generation.configured:
                raise RuntimeError('Set DEEPSEEK_API_KEY in the local .env first')
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(LocalModels(encoder_settings)._path('embedding')),
                                                      local_files_only=True, trust_remote_code=False)
            def count(text):
                return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
            failures = []
            with ThreadPoolExecutor(max_workers=settings.workers) as pool:
                futures = {pool.submit(make_profile, ROOT, record, generation, count,
                    encoder_settings.embedding_max_length): record for record in records}
                for index, future in enumerate(as_completed(futures), 1):
                    record = futures[future]
                    try:
                        profile, cached = future.result()
                        available = [d for d in DIMENSIONS if profile['dimensions'][d]['status'] == 'available']
                        if not cached:
                            print(f"[{index}/{len(records)}] {record['paper']['paper_id']} "
                                  f"extracted {','.join(available)}", flush=True)
                    except Exception as error:
                        failures.append({'paper_id': record['paper']['paper_id'], 'error': str(error)[:1000]})
                        print(f"[{index}/{len(records)}] {record['paper']['paper_id']} FAILED: {error}", flush=True)
                    atomic_json(work / 'progress.json', {'completed': index, 'total': len(records), 'failures': failures})
            if failures:
                raise RuntimeError(f'{len(failures)} paper extractions failed. Completed steps are cached; rerun to resume.')
        if args.stage in {'all', 'graph'}:
            directory, manifest = build_bundle(ROOT, records, settings, encoder_settings,
                                               LocalModels(encoder_settings), snapshot['snapshot_id'])
            print('Published:', directory, flush=True)
            print('Papers:', manifest['papers'], 'API usage (includes reused steps):',
                  manifest['extraction_usage_including_reused_steps'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
