"""Validate and run test50 for the full Docling multimodal retrieval index."""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import sqlite3
from statistics import mean, median
import sys
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.docling_search import DoclingSearch, fuse_objects
from rag.figure_vectors import image_rgb, visual_directory
from rag.storage import atomic_json, read_json
from scripts.docling_prepare import WORK, digest

DATA = ROOT/'data/evaluation/docling_test50/cases.jsonl'
COHORTS = dict(body=8, figure_text=12, table_row=10, number=6, image=6, joint=4, missing_number=4)
FIGURE_ARMS = ('caption', 'caption_explicit', 'caption_context', 'caption_clip', 'all')


def normalized(value):
    return ' '.join(value.split()).casefold()


def validate_gold(db, gold):
    paper = gold['paper_id']
    if gold['kind'] == 'paragraph':
        matches = db.execute("SELECT payload, artifact_dir FROM objects WHERE paper_id=? AND kind='paragraph' "
                             "AND json_extract(payload,'$.ref')=?", (paper,gold['ref'])).fetchall()
    else:
        matches = db.execute('SELECT o.payload,o.artifact_dir FROM labels l JOIN objects o USING(object_id) '
                             'WHERE l.paper_id=? AND l.kind=? AND l.label=?', (paper,gold['kind'],gold['label'])).fetchall()
    candidates = [(json.loads(raw), path) for raw, path in matches]
    candidates = [(obj, path) for obj,path in candidates if gold['page'] in {p['page_no'] for p in obj['provenance']}]
    if len(candidates) != 1:
        raise ValueError(f'Cannot resolve a unique original object: {gold}')
    obj, folder = candidates[0]
    result = {**gold, 'object_id': obj['object_id'], 'pdf_sha256': obj['pdf_sha256']}
    if gold['kind'] == 'paragraph':
        if obj['text'].count(gold['quote']) != 1:
            raise ValueError(f'Gold quote does not occur uniquely: {paper}/{gold["ref"]}')
        start = obj['text'].index(gold['quote'])
        end = start + len(gold['quote'])
        if not any(p['page_no'] == gold['page'] and p['charspan'][0] <= start and end <= p['charspan'][1]
                   for p in obj['provenance']):
            raise ValueError(f'Gold quote is on another page: {paper}/{gold["ref"]}')
        result['span'] = [start, end]
    else:
        crops = [c for c in obj['crops'] if c['page'] == gold['page']]
        if not crops:
            raise ValueError(f'Gold object has no original crop on page {gold["page"]}')
        directory = (ROOT/folder).resolve()
        path = (directory/crops[0]['path']).resolve()
        if not path.is_relative_to(directory) or digest(path) != crops[0]['sha256']:
            raise ValueError('Gold crop checksum mismatch')
        result.update(crop=str(path.relative_to(ROOT)), crop_sha256=crops[0]['sha256'])
    if gold.get('cells'):
        result['structured_gold_checks'] = score_cells(gold['cells'], obj['table']['table_cells'], obj['table']['table_cells'])
    return result


def score_cells(expected, cells, headers):
    out = []
    for item in expected:
        r, c = item['row'], item['col']
        data = {(x['start_row_offset_idx'], x['end_row_offset_idx'], x['start_col_offset_idx'],
                 x['end_col_offset_idx'], x['text']) for x in cells
                if x['start_row_offset_idx'] <= r < x['end_row_offset_idx']
                and x['start_col_offset_idx'] <= c < x['end_col_offset_idx']}
        names = [x['text'] for x in sorted(headers, key=lambda x: x['start_row_offset_idx'])
                 if x.get('column_header') and x['end_row_offset_idx'] <= r
                 and x['start_col_offset_idx'] <= c < x['end_col_offset_idx']]
        header = ' / '.join(names)
        wanted = [normalized(part) for part in item['header'].split(' / ')]
        ordered_names = iter(map(normalized, names))
        out.append({'row': r, 'col': c, 'value_hit': len(data)==1 and normalized(next(iter(data))[4])==normalized(item['value']),
                    'header_hit': all(name in ordered_names for name in wanted), 'returned_header': header})
    return out


def load_set(profile):
    directory = WORK/profile
    snapshot = read_json(directory/'OBJECTS_CURRENT.json')
    validated = read_json(directory/'object-validation.json')
    if (not snapshot or not validated or validated['object_snapshot_id'] != snapshot['snapshot_id']
            or validated['failures'] or snapshot['papers'] != 500):
        raise RuntimeError('The complete, source-validated 500-paper object snapshot is required')
    path = directory/(snapshot['snapshot_id']+'.sqlite3')
    with DATA.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if Counter(x['group'] for x in rows) != COHORTS or len({x['id'] for x in rows}) != 50:
        raise ValueError('Expected the documented 50 distinct cases and cohort sizes')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as db:
        if db.execute('SELECT count(*) FROM papers').fetchone()[0] != snapshot['papers']:
            raise ValueError('Object database paper count mismatch')
        for row in rows:
            if row.get('paper_id') and not db.execute('SELECT 1 FROM papers WHERE paper_id=?', (row['paper_id'],)).fetchone():
                raise ValueError('Query is scoped to a paper absent from the corpus')
            row['gold'] = [validate_gold(db, g) for g in row['gold']]
            if target := row.get('absent'):
                if db.execute('SELECT 1 FROM labels WHERE paper_id=? AND kind=? AND label=?',
                              (row['paper_id'],target['kind'],target['label'])).fetchone():
                    raise ValueError('Negative label already exists')
    return snapshot, rows


def query_image_for(row):
    from PIL import Image, ImageOps
    blob = (ROOT/row['gold'][0]['crop']).read_bytes()
    image = image_rgb(blob)
    if row['transform'] == 'partial_jpeg':
        w, h = image.size
        image = image.crop((round(w*.06), round(h*.08), round(w*.90), round(h*.94)))
    elif row['transform'] == 'resized_padded_jpeg':
        image = ImageOps.expand(image, border=(8, 4, 16, 7), fill='white')
    else:
        raise ValueError('Unknown image transform')
    image.thumbnail((420, 320), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=78)
    return stream.getvalue()


def metrics(hits, gold, key='object_id'):
    wanted = set(gold)
    first = next((i for i,h in enumerate(hits, 1) if h[key] in wanted), None)
    return {'hit1': first==1, 'hit3': first is not None and first<=3, 'hit5': first is not None and first<=5,
            'reciprocal_rank': 1/first if first else 0}


def score_row(row, found, rrf_k):
    scores = {'id': row['id'], 'group': row['group'], 'gold': row['gold']}
    compact_matches = []
    groups = {'paragraph': 'body', 'figure': 'figures', 'table': 'tables'}
    for g in row['gold']:
        hits = found[groups[g['kind']]]
        match = metrics(hits, [g['object_id']])
        if g['kind']=='paragraph':
            start, end = g['span']
            # Require the actual returned windows, not just that the correct paragraph ID appeared.
            cursor = start
            for a,b in sorted(h['view_span'] for h in hits if h['object_id']==g['object_id']):
                if a <= cursor < b:
                    cursor = b
            match['quote_covered'] = cursor >= end
        if g.get('cells'):
            table = next((h for h in hits if h['object_id']==g['object_id']), {})
            rows = table.get('matched_rows', [])
            data = [cell for item in rows for cell in item['cells']]
            values = score_cells(g['cells'], data, table.get('header_cells', []))
            match['row_hit'] = {item['row'] for item in g['cells']} <= {item['row'] for item in rows}
            match['values_hit'] = all(c['value_hit'] for c in values)
            match['values_and_headers_hit'] = all(c['value_hit'] and c['header_hit'] for c in values)
            match['cells'] = values
        compact_matches.append(match)
    scores['required'] = compact_matches
    if row['group'] == 'joint':
        scores['all_evidence'] = all(m.get('quote_covered', m['hit5']) for m in compact_matches)
    if row['group'] == 'number':
        group = groups[row['gold'][0]['kind']]
        scores['exact'] = compact_matches[0]['hit1'] and found[group][0]['exact_number']
    if row['group'] == 'missing_number':
        group = groups[row['absent']['kind']]
        scores['false_exact'] = any(h['exact_number'] for h in found[group])
        scores['fallback_candidates'] = len(found[group])
    if row['group'] == 'figure_text':
        f = found['rankings']['figure']
        caption, context, clip = f.get('caption', []), f.get('context', []), f.get('clip', [])
        explicit = [h for h in context if h.get('via')=='explicit_reference']
        arms = (dict(caption=caption), dict(caption=caption,context=explicit), dict(caption=caption,context=context),
                dict(caption=caption,clip=clip), dict(caption=caption,context=context,clip=clip))
        scores['arms'] = {name: metrics(fuse_objects(arm,rrf_k), [row['gold'][0]['object_id']])
                          for name,arm in zip(FIGURE_ARMS,arms,strict=True)}
        scores['full_branch_order_matches'] = [h['object_id'] for h in fuse_objects(arms[-1], rrf_k)[:5]] == [
            h['object_id'] for h in found['figures']]
        if not scores['full_branch_order_matches']:
            raise RuntimeError('Traced ranks do not reproduce the production figure order')
        scores['figure_branches'] = f
    if row['group'] == 'table_row':
        table = found['rankings']['table']
        scores['table_summary_only'] = metrics(fuse_objects({'table':table['table']},rrf_k), [row['gold'][0]['object_id']])
    scores['returned'] = {k:[{f:h[f] for f in ('object_id','paper_id','pages','exact_number','paths','matched_rows') if f in h}
                            for h in found[k]] for k in ('body','figures','tables')}
    return scores


def summarize(rows):
    def rates(group, fields):
        return {field:mean(row[field] for row in group) for field in fields}

    groups = {}
    for name in COHORTS:
        selected = [r for r in rows if r['group']==name]
        if not selected:
            continue
        g = groups[name] = {'n': len(selected), 'median_seconds': median(r['elapsed_seconds'] for r in selected)}
        keys = ('hit1','hit3','hit5','reciprocal_rank')
        if name == 'joint':
            g.update(rates(selected,['all_evidence']))
        elif name == 'missing_number':
            g.update(rates(selected,['false_exact','fallback_candidates']))
        else:
            extra = {'body':['quote_covered'], 'table_row':['row_hit','values_hit','values_and_headers_hit']}.get(name,[])
            g.update(rates([r['required'][0] for r in selected],[*keys,*extra]))
        if name == 'number':
            g.update(rates(selected,['exact']))
        if name == 'figure_text':
            g['arms'] = {arm: rates([r['arms'][arm] for r in selected],keys) for arm in FIGURE_ARMS}
        if name == 'table_row':
            g['summary_only'] = rates([r['table_summary_only'] for r in selected],keys)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='8bde7f9ad04def90')
    parser.add_argument('--run', action='store_true', help='Run against published full text and CLIP indexes')
    parser.add_argument('--category', choices=COHORTS)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if len(args.profile)!=16 or any(c not in '0123456789abcdef' for c in args.profile) or (args.limit is not None and args.limit<1):
        parser.error('Invalid profile or limit')
    sys.stdout.reconfigure(encoding='utf-8')
    snapshot, rows = load_set(args.profile)
    validation = {'dataset_sha256': digest(DATA), 'object_snapshot_id': snapshot['snapshot_id'], 'profile': args.profile,
                  'cases': len(rows), 'cohorts': COHORTS, 'gold_papers': sorted({g['paper_id'] for r in rows for g in r['gold']}),
                  'image_queries': {row['id']: sha256(query_image_for(row)).hexdigest() for row in rows if row['group']=='image'},
                  'structure_gaps': [{'id': row['id'], 'checks': g['structured_gold_checks']} for row in rows for g in row['gold']
                     if any(not(c['value_hit'] and c['header_hit']) for c in g.get('structured_gold_checks',[]))]}
    atomic_json(WORK/args.profile/'test50-validation.json', validation)
    print(json.dumps(validation,ensure_ascii=False), flush=True)
    if not args.run:
        return
    engine = DoclingSearch(args.profile, body_rerank_k=20)
    if engine.manifest['object_snapshot_id'] != snapshot['snapshot_id'] or len(engine.manifest['documents'])!=500:
        raise RuntimeError('The vector index does not match the complete annotated object snapshot')
    engine.visual()  # Refuse to silently omit the visual arm in the full benchmark.
    visual = read_json(visual_directory(args.profile,False)/'CURRENT.json')
    design = {**validation, 'index_snapshot_id':engine.manifest['snapshot_id'], 'clip_sha256':visual['sha256'],
              'clip_revision':visual['revision'], 'top_k':5, 'category':args.category, 'limit':args.limit,
              'code_sha256':{p:digest(ROOT/p) for p in ('scripts/evaluate_docling.py','rag/docling_search.py','rag/figure_vectors.py')}}
    target = ROOT/'runtime/experiments'/f'docling-test50-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}'
    atomic_json(target/'design.json',design)
    selected = [r for r in rows if args.category is None or r['group']==args.category][:args.limit]
    scores = []
    for row in selected:
        group = row['group']
        kinds = [{'paragraph':'body'}.get(g['kind'],g['kind']) for g in row['gold']] or [row['absent']['kind']]
        started = perf_counter()
        found = engine.search(row['query'] or '', paper_id=row.get('paper_id'), kinds=tuple(set(kinds)), top_k=5,
                             clip_text=row.get('clip_query'), query_image=query_image_for(row) if group=='image' else None,
                             trace=True, query_language='original')
        if group in {'figure_text','image'} and found['visual'] not in {'text','image'}:
            raise RuntimeError('Visual branch became unavailable during evaluation')
        result = score_row(row, found, engine.cfg.rrf_k)
        result['elapsed_seconds'] = round(perf_counter()-started,3)
        atomic_json(target/(row['id']+'.json'),result)
        scores.append(result)
        print(row['id'],group,[m['hit5'] for m in result['required']],result['elapsed_seconds'],flush=True)
    summary = summarize(scores)
    atomic_json(target/'summary.json',{'design':design,'groups':summary,'cases_run':len(scores)})
    print(f'Results: {target / "summary.json"}',flush=True)


if __name__=='__main__':
    main()
