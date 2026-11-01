"""Restore catalog PDFs from their original URLs, checking the recorded SHA-256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restore_paper(paper, root, client, max_bytes=100 * 1024 * 1024):
    pid, expected = paper['paper_id'], paper['sha256']
    if not re.fullmatch(r'\d{4}\.[a-z0-9-]+\.\d+', pid):
        raise ValueError('Invalid catalog paper ID')
    if not re.fullmatch(r'[a-f0-9]{64}', expected):
        raise ValueError('Missing catalog SHA-256: '+pid)
    url = urlparse(paper['pdf_url'])
    if url.scheme != 'https' or url.hostname != 'aclanthology.org':
        raise ValueError('Expected original ACL Anthology HTTPS URL: '+pid)
    target = root/'data/pdfs'/(pid+'.pdf')
    if target.exists():
        if file_sha(target) != expected:
            raise ValueError('Existing PDF differs from the catalog; retain and inspect it: '+pid)
        return {'paper_id':pid, 'status':'cached'}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.'+pid+'-', suffix='.download', delete=False) as stream:
            temporary = Path(stream.name)
            size = 0
            with client.stream('GET', paper['pdf_url']) as response:
                response.raise_for_status()
                for block in response.iter_bytes():
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError('PDF exceeds the download size limit: '+pid)
                    stream.write(block)
        with temporary.open('rb') as stream:
            if stream.read(5) != b'%PDF-':
                raise ValueError('Download is not a PDF: '+pid)
        if file_sha(temporary) != expected:
            raise ValueError('Source PDF changed; recorded annotation hashes were retained: '+pid)
        temporary.replace(target)
        return {'paper_id':pid, 'status':'downloaded'}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Fetch missing PDFs; otherwise only show the selection')
    parser.add_argument('--limit', type=int, default=0, help='0 selects the complete fixed catalog')
    parser.add_argument('--paper-id', nargs='+')
    parser.add_argument('--workers', type=int, choices=range(1,7), default=3)
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    papers = json.loads((ROOT/'data/paper_catalog.json').read_text(encoding='utf-8'))
    if len({p['paper_id'] for p in papers}) != len(papers):
        raise ValueError('Duplicate catalog paper IDs')
    if args.paper_id:
        wanted = set(args.paper_id)
        if wanted - {p['paper_id'] for p in papers}:
            parser.error('Unknown paper ID')
        papers = [p for p in papers if p['paper_id'] in wanted]
    if args.limit:
        papers = papers[:args.limit]
    print(f'Selected {len(papers)} original-source PDFs. Catalog and annotation files are not rewritten.', flush=True)
    if not args.download:
        for p in papers[:5]:
            print(p['paper_id'], p['title'])
        print('Add --download to restore missing PDFs.')
        return 0
    failed = 0
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        def fetch(p):
            try:
                return restore_paper(p,ROOT,client)
            except (httpx.HTTPError,OSError,ValueError) as error:
                return {'paper_id':p['paper_id'],'status':'failed','reason':str(error)}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for row in executor.map(fetch,papers):
                failed += row['status']=='failed'
                print(json.dumps(row,ensure_ascii=False),flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
