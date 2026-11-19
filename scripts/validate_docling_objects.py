"""Validate locally materialized Docling paragraph, figure, and table artifacts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.docling_objects import mentions
from rag.storage import atomic_json, read_json
from scripts.docling_prepare import WORK, artifact, digest, sources, stable_hash


def jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def validate_paper(folder, parsed, pid, pdf_hash, profile):
    info = read_json(folder/'manifest.json')
    if info['paper_id'] != pid or info['pdf_sha256'] != pdf_hash:
        raise ValueError('Paper provenance mismatch')
    for name, expected in info['sha256'].items():
        path = parsed/name if name == 'document.json' else folder/name
        if digest(path) != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    doc = read_json(parsed/'document.json')
    items = {item['self_ref']: item for key in ('texts', 'pictures', 'tables', 'form_items', 'key_value_items')
             for item in doc.get(key, [])}
    objects = {obj['object_id']: obj for obj in jsonl(folder/'objects.jsonl')}
    links, chunks = (jsonl(folder/name) for name in ('links.jsonl', 'chunks.jsonl'))
    if (len(objects), len(links), len(chunks)) != (info['objects'], info['links'], info['chunks']):
        raise ValueError('Artifact count mismatch')
    stats = Counter(papers=1, chunks=len(chunks), links=len(links))
    for obj in objects.values():
        original = items[obj['ref']]
        if (obj['paper_id'], obj['pdf_sha256'], obj['provenance']) != (pid, pdf_hash, original.get('prov', [])):
            raise ValueError('Object differs from its Docling source')
        stats[obj['kind'] + 's'] += 1
        if obj['kind'] == 'paragraph' and obj['text'] != original['text']:
            raise ValueError('Paragraph text differs from Docling source')
        if obj['kind'] == 'table' and obj['table']['table_cells'] != original['data']['table_cells']:
            raise ValueError('Raw table cells differ from Docling source')
        if obj['kind'] in {'table', 'figure'} and not obj['crops']:
            stats['media_without_crop'] += 1
        for crop in obj.get('crops', []):
            path = (folder/crop['path']).resolve()
            if not path.is_relative_to(folder.resolve()) or digest(path) != crop['sha256']:
                raise ValueError('Crop hash or path mismatch')
            if crop['page'] not in {p['page_no'] for p in obj['provenance']}:
                raise ValueError('Crop page does not belong to its object')
            with path.open('rb') as png:
                header = png.read(24)
            if header[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', header[16:]) != (crop['width'], crop['height']):
                raise ValueError('Crop dimensions mismatch')
            stats['crops'] += 1

    reverse = defaultdict(set)
    forward = defaultdict(list)
    for edge in links:
        paragraph, target = (objects[edge[key]] for key in ('paragraph_id', 'object_id'))
        left, right = edge['span']
        if (paragraph['kind'] != 'paragraph' or target['kind'] not in {'figure', 'table'}
                or not 0 <= left < right <= len(paragraph['text'])):
            raise ValueError('Invalid graph endpoints or text range')
        if not any((kind, label, a, b) == (target['kind'], edge['label'], left, right)
                   for kind, label, a, b in mentions(paragraph['text'])):
            raise ValueError('Link range does not identify its numbered mention')
        forward[edge['paragraph_id']].append(edge)
        reverse[edge['object_id']].add(edge['paragraph_id'])

    spans = defaultdict(list)
    for chunk in chunks:
        obj = objects[chunk['object_id']]
        if not 1 <= chunk['tokens'] <= profile['options_chunker']['max_tokens'] or not chunk['pages']:
            raise ValueError('Chunk has invalid token budget or no pages')
        prov_pages = sorted({p['page_no'] for p in obj['provenance'] if str(p['page_no']) in doc['pages']})
        if chunk['pages'] != prov_pages:
            raise ValueError('Chunk page provenance differs from its source object')
        left, right = chunk['view_span']
        if chunk['view'] == 'paragraph':
            if chunk['text'] != obj['text'][left:right]:
                raise ValueError('Paragraph chunk differs from its source range')
            spans[obj['object_id']].append((left, right))
        elif chunk['view'] == 'table_row' and chunk['row'] not in {row['row'] for row in obj['rows']}:
            raise ValueError('Table row chunk points to a missing row')
        linked = {e['object_id'] for e in forward[obj['object_id']] if left < e['span'][1] and right > e['span'][0]}
        if (set(chunk['linked_object_ids']) != linked
                or set(chunk['linked_paragraph_ids']) != reverse[obj['object_id']]):
            raise ValueError('Chunk graph links differ from the object graph')

    for obj in objects.values():
        if obj['kind'] != 'paragraph' or not obj['provenance']:
            continue
        cursor = 0
        for left, right in sorted(spans[obj['object_id']]):
            if left > cursor and obj['text'][cursor:left].strip():
                raise ValueError('Chunking lost paragraph text between windows')
            cursor = max(cursor, right)
        if obj['text'][cursor:].strip():
            raise ValueError('Chunking lost the end of a paragraph')
        if len(spans[obj['object_id']]) > 1:
            stats['paragraphs_split_into_windows'] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--require-full', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-f0-9]{16}', args.profile):
        parser.error('Invalid Docling object profile ID')
    directory = WORK/args.profile
    profile = read_json(directory/'profile.json')
    if not profile or stable_hash(profile)[:16] != args.profile or profile['format_version'] < 3:
        raise ValueError('Invalid object profile')
    base, _ = sources()
    totals, failures, inspected = Counter(), [], 0
    for pid, sha in sorted(base['documents'].items()):
        folder = artifact(args.profile, pid, sha)
        if not (folder/'manifest.json').is_file():
            continue
        inspected += 1
        try:
            totals.update(validate_paper(folder, artifact('parsed/'+profile['parse_id'], pid, sha), pid, sha, profile))
        except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
            failures.append({'paper_id': pid, 'error': str(error)})
    if not inspected:
        raise ValueError('No materialized papers found')
    full = read_json(directory/'OBJECTS_CURRENT.json')
    full_verified = (inspected == len(base['documents']) and not failures and full
                     and full['baseline_snapshot'] == base['snapshot_id'] and full['papers'] == inspected)
    report = {'profile': args.profile, 'baseline_snapshot': base['snapshot_id'], 'checked_papers': inspected,
              'expected_papers': len(base['documents']), 'stats': dict(totals), 'failures': failures,
              'object_snapshot_id': full['snapshot_id'] if full_verified else None}
    atomic_json(directory/'object-validation.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    raise SystemExit(bool(failures) or (args.require_full and not full_verified))


if __name__ == '__main__':
    main()
