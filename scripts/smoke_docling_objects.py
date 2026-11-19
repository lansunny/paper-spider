"""Exercise an isolated vector pilot before allowing the full Docling build."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.docling_index import build, object_snapshot, search, settings_for
from rag.embeddings import LocalModels
from rag.storage import atomic_json, read_json
from scripts.docling_prepare import WORK, sources

PAPERS = ('2020.emnlp-main.550', '2021.emnlp-main.224', '2023.acl-long.557')
IRCOT = PAPERS[-1]


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def check_hits(result, manifest, db):
    require(result['snapshot_id'] == manifest['snapshot_id'] and result['object_db'] == manifest['object_db'],
            'Search returned a different pilot snapshot')
    require(bool(result['hits']), 'No search hits')
    for hit in result['hits']:
        paper = hit['paper_id']
        require(manifest['documents'].get(paper) == hit['document_sha256'] and bool(hit['pages']),
                'Search hit escaped its pilot PDFs or has no page')
        obj = result['objects'][hit['object_id']]
        require(obj['paper_id'] == paper and obj['pdf_sha256'] == hit['document_sha256'], 'Object provenance mismatch')
        for item in hit['related']:
            require(item['paper_id'] == paper and bool(item['pages']), 'Linked evidence escaped its original PDF')
        for crop in obj['crops']:
            require((ROOT/crop['path']).is_file() and crop['page'] in obj['source_pages'], 'Missing crop or invalid crop page')
        if hit.get('table_row'):
            row = hit['table_row']
            payload = db.execute('SELECT payload FROM objects WHERE object_id=?', (hit['object_id'],)).fetchone()[0]
            parsed = json.loads(payload)
            require(row['cells'] == [parsed['table']['table_cells'][i] for i in row['cell_ids']], 'Returned row cells differ from object')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--existing-pilot', action='store_true', help='Verify the smoke harness without authorizing a full build')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    profile, cfg = settings_for(args.profile, pilot=True)
    require(profile['format_version'] >= 3, 'Smoke test requires an object corpus')
    baseline, _ = sources()
    object_db, snapshot = object_snapshot(args.profile, PAPERS, baseline, pilot=True, from_full=not args.existing_pilot)
    directory = WORK/args.profile
    report_path = directory/('existing-pilot-smoke.json' if args.existing_pilot else 'object-smoke.json')
    report = {'profile': args.profile, 'baseline_snapshot': baseline['snapshot_id'], 'object_snapshot_id': snapshot,
              'paper_ids': list(PAPERS), 'status': 'running', 'checks': []}
    atomic_json(report_path, report)
    try:
        if not args.existing_pilot:
            validated = read_json(directory/'object-validation.json') or {}
            require(validated.get('object_snapshot_id') == snapshot and validated.get('failures') == []
                    and validated.get('checked_papers') == len(baseline['documents']), 'Full object validation is missing')
        manifest = build(args.profile, paper_ids=PAPERS, from_full_objects=not args.existing_pilot)
        require(manifest['object_db'] == object_db and manifest['object_snapshot_id'] == snapshot,
                'Pilot does not use the intended object snapshot')
        report['pilot_snapshot_id'] = manifest['snapshot_id']
        model = LocalModels(cfg)
        with closing(sqlite3.connect((ROOT/object_db).as_uri()+'?mode=ro', uri=True)) as db:
            result = search(args.profile, 'DPR in-batch negatives BM25 hard negatives', pilot=True, models=model)
            check_hits(result, manifest, db)
            require(any(h['paper_id'] == PAPERS[0] and h['kind'] == 'text' for h in result['hits']),
                    'DPR prose absent from top five search hits')
            report['checks'].append('body_hybrid_search')

            for label, kind, page in [('Figure 2', 'figure', 4), ('Table 1', 'table', 8)]:
                result = search(args.profile, 'IRCoT '+label, paper_id=IRCOT, pilot=True, models=model)
                check_hits(result, manifest, db)
                top = result['hits'][0]
                require(top.get('retrieval_origin') == 'object_number' and top['kind'] == kind and page in top['pages']
                        and result['objects'][top['object_id']]['crops'], f'{label} did not locate the expected original crop')
                report['checks'].append(kind+'_number_and_crop')
                if kind == 'table':
                    citing = [item for item in top['related'] if item['kind'] == 'paragraph']
                    require(bool(citing), 'Table 1 has no citing paragraph')
                    link = db.execute('SELECT 1 FROM links WHERE paragraph_id=? AND object_id=?',
                                      (citing[0]['object_id'], top['object_id'])).fetchone()
                    require(link is not None, 'Table-to-paragraph evidence has no persistent reverse link')
                    report['checks'].append('table_paragraph_links')
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        atomic_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
