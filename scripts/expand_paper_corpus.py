"""Expand the learning corpus from official ACL metadata and validated PDFs.

Run with the project interpreter: scripts/expand_paper_corpus.py --target 200.
Discovery is cached; successful downloads are checkpointed for safe resumption.
"""
import argparse
import base64
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET

import httpx
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.storage import atomic_json, read_json

COLLECTIONS = ['2025.acl', '2025.emnlp', '2025.naacl', '2024.acl',
               '2024.emnlp', '2023.emnlp']
TOPICS = {
    'rag': r'retrieval[- ]augment|\brag\b|\w+rag\b|retrieval[- ]enhanced|retrieval[- ]generation',
    'retrieval': r'dense retrieval|sparse retrieval|hybrid retrieval|passage retrieval|document retrieval|text retrieval|neural retrieval|retriever|re[- ]?rank|text embeddings?|sentence embeddings?',
    'evaluation': r'hallucinat|faithful|factual|attribution|citation|grounded|grounding',
    'long_context': r'long[- ]context|long[- ]document|multi[- ]document|context compression|context selection|document chunk|chunking|multi[- ]hop|multihop',
    'scientific_qa': r'scientific.*(?:question|answer|document|paper|literature)|(?:question|answer).*scientific|research papers|scholarly|literature review|paperqa|qasper',
}
WEIGHTS = {'rag': 7, 'retrieval': 4, 'evaluation': 3, 'long_context': 2, 'scientific_qa': 2}
HTTP = httpx.Client(timeout=60, follow_redirects=True,
                    limits=httpx.Limits(max_connections=6, max_keepalive_connections=6),
                    headers={'User-Agent': 'PaperRAG-learning-corpus/1.0'})


def normalized_title(title):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKC', title).lower())


def get(url, *, max_bytes=100 * 1024 * 1024):
    for attempt in range(3):
        try:
            with HTTP.stream('GET', url) as response:
                response.raise_for_status()
                data = bytearray()
                for part in response.iter_bytes():
                    data.extend(part)
                    if len(data) > max_bytes:
                        raise ValueError('Download exceeds size budget')
                return bytes(data)
        except (httpx.HTTPError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def metadata(collection):
    directory = ROOT / 'data/import/acl_metadata'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (collection + '.xml')
    provenance_path = path.with_suffix('.source.json')
    if not path.exists() or not provenance_path.exists():
        url = f'https://api.github.com/repos/acl-org/acl-anthology/contents/data/xml/{collection}.xml'
        meta = json.loads(get(url))
        blob = json.loads(get(meta['git_url']))
        data = base64.b64decode(blob['content'])
        digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if digest != meta['sha']:
            raise ValueError('ACL metadata Git blob hash mismatch')
        ET.fromstring(data)
        path.write_bytes(data)
        atomic_json(provenance_path, {'source_url': meta['html_url'], 'git_blob_sha': digest,
                                     'sha256': hashlib.sha256(data).hexdigest()})
    source = read_json(provenance_path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != source['sha256']:
        raise ValueError('Cached metadata hash mismatch')
    root = ET.fromstring(data)
    records = []
    for volume in root.findall('volume'):
        year = int(volume.findtext('meta/year'))
        for paper in volume.findall('paper'):
            title = ''.join(paper.find('title').itertext()).strip()
            tags = [topic for topic, pattern in TOPICS.items() if re.search(pattern, title, re.I)]
            if not tags or paper.find('pdf') is None:
                continue
            doi = paper.findtext('doi', '')
            identifier = doi.removeprefix('10.18653/v1/')
            if not re.fullmatch(r'\d{4}\.[a-z0-9-]+\.\d+', identifier):
                continue
            authors = [' '.join(filter(None, [a.findtext('first'), a.findtext('last')]))
                       for a in paper.findall('author')]
            abstract_node = paper.find('abstract')
            abstract = ''.join(abstract_node.itertext()).strip() if abstract_node is not None else ''
            records.append({'id': f'https://aclanthology.org/{identifier}/', 'paper_id': identifier,
                            'title': title, 'citation_info': ', '.join(authors), 'abstract': abstract,
                            'publication_year': year, 'doi': doi, 'source': 'ACL Anthology',
                            'landing_url': f'https://aclanthology.org/{identifier}/',
                            'pdf_url': f'https://aclanthology.org/{identifier}.pdf',
                            'topics': tags, 'metadata_source': source})
    print(json.dumps({'metadata': collection, 'relevant_candidates': len(records)}), flush=True)
    return records


def ranked_candidates(records, existing):
    ids = {p['paper_id'] for p in existing}
    titles = {normalized_title(p['title']) for p in existing}
    buckets = defaultdict(list)
    for paper in sorted(records, key=lambda p: p['paper_id']):
        title_text = paper['title'].lower()
        # Keep the corpus about textual/scientific retrieval, not arbitrary uses
        # of 'grounding', 'ranking', or retrieval in unrelated modalities.
        if re.search(r'visual instruction|image caption|visual program|robotics|video|spatial|map grounding|visual storytelling|object hallucination|visual grounding|time series|product matching|machine translation|keyphrase|schematic diagram|discharge summary|named entity|\bner\b|entity link|clinical predictions|knowledge editing|safety alignment|few-shot classification|preference re-ranking', title_text):
            continue
        title = normalized_title(paper['title'])
        if paper['paper_id'] in ids or title in titles:
            continue
        ids.add(paper['paper_id'])
        titles.add(title)
        # Prefer distinctive topics before the broad RAG label.
        primary = next(t for t in ['scientific_qa', 'long_context', 'rag', 'retrieval', 'evaluation']
                       if t in paper['topics'])
        abstract = paper.get('abstract', '').lower()
        relevance = sum(bool(re.search(pattern, title_text)) * weight for pattern, weight in [
            (r'retrieval[- ]augmented|\brag\b|\w+rag\b', 8),
            (r'question answering|scientific.*(?:question|paper|literature)|literature synthesis|scholar qa', 5),
            (r'dense|sparse|passage retrieval|document retrieval|text re-ranking|rerank|reranking|chunk|compression', 4),
            (r'faithful|citation|attribut|evidence', 3),
        ])
        relevance += 2 * sum(term in abstract for term in ['retrieval-augmented', 'retrieval augmented', 'question answering', 'retriever'])
        buckets[primary].append({**paper, 'primary_topic': primary, 'selection_score': relevance})
    queues = {}
    for topic, papers in buckets.items():
        years = defaultdict(deque)
        for paper in sorted(papers, key=lambda p: (-p['selection_score'], -len(p['topics']), p['paper_id'])):
            years[paper['publication_year']].append(paper)
        balanced = deque()
        while any(years.values()):
            for year in sorted(years, reverse=True):
                if years[year]:
                    balanced.append(years[year].popleft())
        queues[topic] = balanced
    result = []
    while any(queues.values()):
        for topic, weight in WEIGHTS.items():
            for _ in range(weight):
                if queues.get(topic):
                    result.append(queues[topic].popleft())
    return result


def download(paper):
    identifier = paper['paper_id']
    path = ROOT / 'data/pdfs' / (identifier + '.pdf')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        content = path.read_bytes() if path.exists() else get(paper['pdf_url'])
        if not content.startswith(b'%PDF-'):
            raise ValueError('Response is not PDF')
        with pymupdf.open(stream=content, filetype='pdf') as pdf:
            if pdf.needs_pass or len(pdf) < 3:
                raise ValueError('Encrypted or too short')
            chars = sum(len(page.get_text().strip()) for page in pdf)
            if chars < 1500:
                raise ValueError('Insufficient extractable text')
            pages = len(pdf)
        if not path.exists():
            partial = path.with_suffix('.pdf.part')
            partial.write_bytes(content)
            partial.replace(path)
        result = {**paper, 'status': 'downloaded', 'pdf_path': path.relative_to(ROOT).as_posix(),
                  'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content),
                  'pdf_pages': pages, 'extractable_chars': chars,
                  'retrieved_on': datetime.now().astimezone().date().isoformat()}
        page_path = ROOT / 'data/source_pages' / (identifier + '.html')
        page_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            html_bytes = page_path.read_bytes() if page_path.exists() else get(paper['landing_url'])
            if not page_path.exists():
                page_path.write_bytes(html_bytes)
            links = sorted(set(re.findall(r'https?://creativecommons.org/licenses/[a-zA-Z0-9./-]+',
                                          html_bytes.decode('utf-8'))))
            result.update(license_links=links, source_page_sha256=hashlib.sha256(html_bytes).hexdigest(),
                          license_status='source_page_links_recorded' if links else 'not_found_review_before_redistribution')
        except Exception as error:
            result.update(license_status='source_page_unavailable', source_page_error=type(error).__name__)
        return result
    except Exception as error:
        return {**paper, 'status': 'failed', 'error': f'{type(error).__name__}: {error}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=int, default=200)
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=3)
    parser.add_argument('--discover-only', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'data/paper_manifest.json'
    existing = read_json(path, [])
    if any(p.get('status') != 'downloaded' for p in existing):
        raise ValueError('Existing manifest contains incomplete entries; inspect before expanding')
    if len(existing) >= args.target:
        print(json.dumps({'ready': len(existing), 'target': args.target}))
        return
    candidates_path = ROOT / 'data/import/expansion_candidates.json'
    if candidates_path.exists():
        records = read_json(candidates_path)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            records = [p for group in pool.map(metadata, COLLECTIONS) for p in group]
        atomic_json(candidates_path, records)
    candidates = ranked_candidates(records, existing)
    print(json.dumps({'existing': len(existing), 'candidates': len(candidates),
                      'topics': dict(Counter(p['primary_topic'] for p in candidates))}), flush=True)
    if args.discover_only:
        atomic_json(ROOT / 'data/import/expansion_selection.json', candidates[:args.target - len(existing)])
        return
    backup = ROOT / 'data/import/corpus_versions' / hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    backup.mkdir(parents=True, exist_ok=True)
    for name in ['paper_manifest.json', 'paper_catalog.json']:
        original = ROOT / 'data' / name
        if original.exists() and not (backup / name).exists():
            (backup / name).write_bytes(original.read_bytes())
    hashes = {p['sha256'] for p in existing}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cursor, pending = 0, set()
        while len(existing) < args.target:
            # A slow PDF must not leave the other download workers idle.
            # Limit in-flight work so successful downloads never exceed target.
            while (len(pending) < min(args.workers, args.target - len(existing))
                   and cursor < len(candidates)):
                pending.add(pool.submit(download, candidates[cursor]))
                cursor += 1
            if not pending:
                break
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                paper = future.result()
                if paper['status'] == 'downloaded' and paper['sha256'] not in hashes:
                    existing.append(paper)
                    hashes.add(paper['sha256'])
                    atomic_json(path, existing)
                    atomic_json(ROOT / 'data/paper_catalog.json', existing)
                else:
                    failures.append({'paper_id': paper['paper_id'], 'error': paper.get('error', 'duplicate_pdf_hash')})
                print(json.dumps({'ready': len(existing), 'target': args.target, 'paper': paper['paper_id'],
                                  'status': paper['status']}), flush=True)
    summary = {'papers': len(existing), 'target': args.target, 'failures': failures,
               'years': dict(Counter(str(p.get('publication_year')) for p in existing)),
               'added_topics': dict(Counter(p.get('primary_topic', 'original') for p in existing)),
               'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    atomic_json(ROOT / 'runtime/corpus_expansion_summary.json', summary)
    print(json.dumps(summary), flush=True)
    if len(existing) < args.target:
        raise SystemExit('Not enough validated PDFs to meet target')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        main()
    finally:
        HTTP.close()
