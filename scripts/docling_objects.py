"""Materialize the versioned paragraph / figure / table corpus from saved Docling parses."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re
import sqlite3
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.docling_objects import RENDER_SCALE, materialize
from rag.storage import atomic_json, atomic_text, read_json
from scripts.docling_prepare import BASE, BGE, WORK, artifact, digest, sources, stable_hash


def jsonl(items):
    return ''.join(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n' for item in items)


def database(directory, manifest, docs):
    path = directory / (manifest['snapshot_id'] + '.sqlite3')
    tmp = path.with_suffix('.tmp')
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        con.executescript('''
            PRAGMA foreign_keys=ON;
            CREATE TABLE papers(paper_id TEXT PRIMARY KEY, pdf_sha256 TEXT NOT NULL, title TEXT NOT NULL,
                                pdf_path TEXT NOT NULL, docling_path TEXT NOT NULL);
            CREATE TABLE objects(object_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL REFERENCES papers,
                                  kind TEXT NOT NULL, ordinal INTEGER NOT NULL, artifact_dir TEXT NOT NULL,
                                  payload TEXT NOT NULL);
            CREATE TABLE links(paragraph_id TEXT NOT NULL REFERENCES objects, object_id TEXT NOT NULL REFERENCES objects,
                                start INTEGER NOT NULL, end INTEGER NOT NULL, label TEXT NOT NULL,
                                PRIMARY KEY(paragraph_id, object_id, start, end));
            CREATE TABLE labels(paper_id TEXT NOT NULL REFERENCES papers, kind TEXT NOT NULL,
                                label TEXT NOT NULL, object_id TEXT NOT NULL REFERENCES objects,
                                PRIMARY KEY(paper_id,kind,label,object_id));
            CREATE INDEX objects_by_paper ON objects(paper_id,kind,ordinal);
            CREATE INDEX links_by_object ON links(object_id);
        ''')
        with con:
            for pid, (path_to_pdf, pdf_hash, title, docling, folder) in docs.items():
                con.execute('INSERT INTO papers VALUES(?,?,?,?,?)', (pid, pdf_hash, title, path_to_pdf, docling))
                for line in (folder/'objects.jsonl').open(encoding='utf-8'):
                    obj = json.loads(line)
                    con.execute('INSERT INTO objects VALUES(?,?,?,?,?,?)', (obj['object_id'], pid, obj['kind'],
                                obj['ordinal'], folder.relative_to(ROOT).as_posix(), line.rstrip('\n')))
                    con.executemany('INSERT INTO labels VALUES(?,?,?,?)', [(pid, obj['kind'], n, obj['object_id'])
                                                                          for n in obj.get('labels', [])])
                with (folder/'links.jsonl').open(encoding='utf-8') as stream:
                    for line in stream:
                        e = json.loads(line)
                        con.execute('INSERT INTO links VALUES(?,?,?,?,?)', (e['paragraph_id'], e['object_id'],
                                                                           *e['span'], e['label']))
        if con.execute('PRAGMA quick_check').fetchone() != ('ok',) or con.execute('PRAGMA foreign_key_check').fetchone():
            raise RuntimeError('Object database integrity check failed')
        counts = {table: con.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in ('papers', 'objects', 'links')}
        if counts != {key: manifest[key] for key in counts}:
            raise RuntimeError('Object database count mismatch')
    finally:
        con.close()
    tmp.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parse-profile', required=True, help='Profile ID reported by docling_prepare.py')
    parser.add_argument('--paper-id', nargs='+')
    parser.add_argument('--cache-ready', action='store_true', help='Prepare only already parsed papers; do not publish a snapshot')
    parser.add_argument('--max-tokens', type=int, default=384)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--profile-out', type=Path, help='Write the successfully published object manifest here')
    args = parser.parse_args()
    if (not re.fullmatch(r'[a-f0-9]{16}', args.parse_profile) or not 64 <= args.max_tokens <= 510 or
            not 0 <= args.overlap < args.max_tokens//2 or (args.cache_ready and args.profile_out)):
        parser.error('Invalid profile, max-tokens or overlap')
    import pymupdf
    from transformers import AutoTokenizer

    sys.stdout.reconfigure(encoding='utf-8')
    baseline, catalog = sources()
    ids = sorted(args.paper_id or catalog)
    if len(ids) != len(set(ids)) or not set(ids).issubset(catalog):
        parser.error('Expected unique paper IDs from the active ACL baseline')
    base_profile = read_json(WORK/args.parse_profile/'profile.json')
    if not base_profile or stable_hash(base_profile)[:16] != args.parse_profile:
        parser.error('Unknown Docling preparation profile')
    profile = {k: base_profile[k] for k in ('parser', 'parse_id', 'bge', 'tokenizer_sha256')}
    profile.update(format_version=3, options_chunker={'algorithm': 'paragraph-overlap-v1', 'max_tokens': args.max_tokens,
                                                      'overlap': args.overlap},
                   objects={'version': 'docling-objects-v2', 'crop_scale': RENDER_SCALE, 'pymupdf': version('pymupdf')})
    if digest(BGE/'tokenizer.json') != profile['tokenizer_sha256'] or read_json(BGE/'paperrag-model.json') != profile['bge']:
        raise ValueError('BGE tokenizer or model has changed')
    profile_id = stable_hash(profile)[:16]
    directory = WORK/profile_id
    atomic_json(directory/'profile.json', profile)
    if args.cache_ready:
        ids = [pid for pid in ids if (artifact('parsed/'+profile['parse_id'], pid,
                                               baseline['documents'][pid])/'manifest.json').is_file()]
        if not ids:
            raise ValueError('No cached Docling parses are ready')
    print(f'Corpus {baseline["snapshot_id"]}; object profile {profile_id}; {len(ids)} papers', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BGE, local_files_only=True)
    rows, failures, docs = [], [], {}
    for i, pid in enumerate(ids, 1):
        start = perf_counter()
        entry = catalog[pid]
        pdf = (ROOT/entry['pdf_path']).resolve()
        pdf_hash = baseline['documents'][pid]
        parsed = artifact('parsed/'+profile['parse_id'], pid, pdf_hash)
        folder = artifact(profile_id, pid, pdf_hash)
        source = parsed/'document.json'
        try:
            if not pdf.is_relative_to(ROOT.resolve()) or digest(pdf) != pdf_hash:
                raise ValueError('PDF does not match the baseline version')
            parsed_meta = read_json(parsed/'manifest.json')
            if not parsed_meta or digest(source) != parsed_meta['sha256']:
                raise ValueError('No matching Docling document')
            info = read_json(folder/'manifest.json')
            cache_files = ('objects.jsonl', 'links.jsonl', 'chunks.jsonl')
            cached = (info and info['sha256']['document.json'] == parsed_meta['sha256'] and
                      all((folder/f).exists() and digest(folder/f) == info['sha256'][f] for f in cache_files))
            if cached:
                with (folder/'objects.jsonl').open(encoding='utf-8') as stream:
                    for line in stream:
                        if any(not (folder/crop['path']).is_file() or digest(folder/crop['path']) != crop['sha256']
                               for crop in json.loads(line).get('crops', [])):
                            cached = False
                            break
            if not cached:
                document = read_json(source)
                with pymupdf.open(pdf) as original:
                    if len(original) != len(document['pages']):
                        raise ValueError('Docling and PDF page counts differ')
                    objects, links, chunks, notices = materialize(document, original, folder, pid, pdf_hash,
                        profile['parse_id'], entry['title'], tokenizer, args.max_tokens, args.overlap)
                if not chunks:
                    raise ValueError('No indexable views')
                if digest(pdf) != pdf_hash:
                    raise ValueError('PDF changed during materialization')
                data = {'objects.jsonl': objects, 'links.jsonl': links, 'chunks.jsonl': chunks}
                for name, values in data.items():
                    atomic_text(folder/name, jsonl(values))
                info = {'profile_id': profile_id, 'pdf_sha256': pdf_hash, 'paper_id': pid,
                        'chunks': len(chunks), 'objects': len(objects), 'links': len(links), 'notices': notices,
                        'chunk_kinds': dict(Counter(c['kind'] for c in chunks)),
                        'object_kinds': dict(Counter(o['kind'] for o in objects)),
                        'crops': sum(len(o.get('crops', [])) for o in objects),
                        'sha256': {'document.json': parsed_meta['sha256'], **{f: digest(folder/f) for f in cache_files}}}
                atomic_json(folder/'manifest.json', info)
            rows.append(info)
            docs[pid] = (entry['pdf_path'], pdf_hash, entry['title'], source.relative_to(ROOT).as_posix(), folder)
            print(f'[{i}/{len(ids)}] {pid}: {info["objects"]} objects, {info["links"]} links, {info["chunks"]} chunks '
                  f'({"cached" if cached else f"{perf_counter()-start:.1f}s"})', flush=True)
        except Exception as error:
            failures.append({'paper_id': pid, 'type': type(error).__name__, 'message': str(error)[:400]})
            print(f'[{i}/{len(ids)}] {pid}: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
    name = ('objects-ready' if args.cache_ready else
            'objects-full' if len(ids) == len(catalog) else 'objects-subset-'+stable_hash(ids)[:8])
    report = {'baseline_snapshot': baseline['snapshot_id'], 'profile_id': profile_id, 'requested': len(ids),
              'completed': len(rows), 'failures': failures, 'papers': rows}
    atomic_json(directory/(name+'.json'), report)
    if failures:
        raise SystemExit(1)
    if args.cache_ready:
        return
    if read_json(BASE)['snapshot_id'] != baseline['snapshot_id']:
        raise ValueError('Baseline snapshot changed during materialization')
    snapshot = stable_hash([profile_id, [(x['paper_id'], x['sha256']) for x in rows]])[:32]
    manifest = {'profile_id': profile_id, 'snapshot_id': snapshot, 'papers': len(rows),
                'baseline_snapshot': baseline['snapshot_id'], 'built_at': datetime.now(timezone.utc).isoformat(),
                **{k: sum(row[k] for row in rows) for k in ('objects', 'links', 'chunks', 'crops')}}
    db = database(directory, manifest, docs)
    manifest['database'] = db.name
    atomic_json(directory/('OBJECTS_CURRENT.json' if len(ids) == len(catalog) else 'OBJECTS_PILOT.json'), manifest)
    if args.profile_out:
        atomic_json(args.profile_out, manifest)
    print({**manifest, 'database': str(db)}, flush=True)


if __name__ == '__main__':
    main()
