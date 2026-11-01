"""Download curated ACL PDFs, preserve source pages and checksums."""
import concurrent.futures
import hashlib
import json
import re
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    result = subprocess.run(['curl.exe', '--silent', '--show-error', '--fail', '--location', '--retry', '1', '--connect-timeout', '15', '--max-time', '240', url, '--output', str(partial)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-500:])
    partial.replace(destination)

def download(paper):
    identifier = paper['paper_id']
    output = ROOT / 'data/pdfs' / (identifier + '.pdf')
    page = ROOT / 'data/source_pages' / (identifier + '.html')
    landing_url = 'https://aclanthology.org/' + identifier + '/'
    result = {**paper, 'landing_url': landing_url, 'retrieved_on': '2026-10-01'}
    try:
        if not output.exists():
            fetch(paper['pdf_url'], output)
        content = output.read_bytes()
        if not content.startswith(b'%PDF-'):
            raise ValueError('Response is not a PDF')
        result.update(status='downloaded', pdf_path=output.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    except Exception as error:
        result.update(status='failed', error=str(error))
    try:
        if not page.exists():
            fetch(landing_url, page)
        html = page.read_text(encoding='utf-8')
        licenses = sorted(set(re.findall(r'https?://creativecommons.org/licenses/[a-zA-Z0-9./-]+', html)))
        result.update(license_links=licenses, license_status='source_page_links_recorded' if licenses else 'not_found_review_before_redistribution')
        result['source_page_sha256'] = hashlib.sha256(page.read_bytes()).hexdigest()
    except Exception as error:
        result.update(license_status='source_page_unavailable', source_page_error=str(error))
    print(json.dumps({'paper': identifier, 'status': result['status'], 'license': result['license_status']}), flush=True)
    return result

if __name__ == '__main__':
    papers = json.loads((ROOT / 'data/paper_catalog.json').read_text(encoding='utf-8'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, papers))
    (ROOT / 'data/paper_manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    failures = [paper['paper_id'] for paper in results if paper['status'] != 'downloaded']
    print(json.dumps({'downloaded': len(results) - len(failures), 'failed': failures}))
    if failures:
        raise SystemExit(1)
