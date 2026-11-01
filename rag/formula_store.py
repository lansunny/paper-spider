"""One explicit active full-document extraction per paper; no directory guessing."""
from pathlib import Path

import portalocker

from rag.config import ROOT
from rag.formula_schema import FormulaElement
from rag.storage import atomic_json, read_json


def digest_file(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_directory(directory, root):
    directory = Path(directory).resolve()
    if not directory.is_relative_to((root/'runtime/formulas').resolve()):
        raise ValueError('Formula artifact directory must remain under runtime/formulas')
    return directory


def publish_document(directory, root=ROOT):
    """Only a complete, validated full-paper run can become authoritative."""
    directory = checked_directory(directory, root)
    manifest = read_json(directory/'manifest.json')
    if (not manifest or manifest.get('status') != 'completed'
            or manifest.get('selected_pages') != list(range(1, manifest.get('page_count', 0)+1))
            or not manifest.get('page_count') or not manifest.get('canonical_pages_sha256')):
        raise ValueError('Only a completed full-document extraction can be published')
    for name, key in [('elements.jsonl', 'elements_sha256'), ('canonical_pages.jsonl', 'canonical_pages_sha256')]:
        if digest_file(directory/name) != manifest[key]:
            raise ValueError('Formula artifact integrity check failed')
    elements = [FormulaElement.model_validate_json(s) for s in (directory/'elements.jsonl').read_text(encoding='utf-8').splitlines() if s]
    if len(elements) != manifest['elements']:
        raise ValueError('Formula extraction count mismatch')
    for e in elements:
        asset = (root/e.image_path).resolve()
        if (e.paper_id != manifest['paper_id'] or e.document_sha256 != manifest['document_sha256']
                or not asset.is_relative_to(directory) or digest_file(asset) != e.image_sha256):
            raise ValueError('Formula image is missing, changed or belongs to another extraction')
    pdf = (root/manifest['pdf_path']).resolve()
    if not pdf.is_relative_to(root.resolve()) or digest_file(pdf) != manifest['document_sha256']:
        raise ValueError('Source PDF no longer matches extraction')
    path = root/'runtime/formulas/CURRENT.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'directory': directory.relative_to(root).as_posix(),
              'document_sha256': manifest['document_sha256'],
              'manifest_sha256': digest_file(directory/'manifest.json'),
              'extractor_version': manifest['profile']['version'],
              'pages': manifest['selected_pages'], 'elements': manifest['elements']}
    with portalocker.Lock(str(path)+'.lock', timeout=10):
        current = read_json(path, {'schema_version': 1, 'papers': {}})
        current['papers'][manifest['paper_id']] = record
        atomic_json(path, current)
    return record


class FormulaStore:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.records = read_json(self.root/'runtime/formulas/CURRENT.json', {'papers': {}})['papers']
        self._loaded = {}

    def document(self, paper_id, document_sha256=None):
        record = self.records.get(paper_id)
        if not record or (document_sha256 is not None and record['document_sha256'] != document_sha256):
            return None
        if paper_id not in self._loaded:
            directory = checked_directory(self.root/record['directory'], self.root)
            if digest_file(directory/'manifest.json') != record['manifest_sha256']:
                raise ValueError('Active formula manifest was changed')
            manifest = read_json(directory/'manifest.json')
            if manifest['document_sha256'] != record['document_sha256'] or manifest['paper_id'] != paper_id:
                raise ValueError('Active formula source identity mismatch')
            source_pdf = (self.root/manifest['pdf_path']).resolve()
            if not source_pdf.is_relative_to(self.root) or digest_file(source_pdf) != record['document_sha256']:
                raise ValueError('Active formula source PDF changed')
            for name, key in [('elements.jsonl', 'elements_sha256'), ('canonical_pages.jsonl', 'canonical_pages_sha256')]:
                if digest_file(directory/name) != manifest[key]:
                    raise ValueError('Active formula artifacts were changed')
            elements = [FormulaElement.model_validate_json(s) for s in (directory/'elements.jsonl').read_text(encoding='utf-8').splitlines() if s]
            import json
            pages = {p['pdf_page']: p for p in (json.loads(s) for s in (directory/'canonical_pages.jsonl').read_text(encoding='utf-8').splitlines() if s)}
            if len(elements) != manifest['elements'] or sorted(pages) != manifest['selected_pages']:
                raise ValueError('Active formula artifacts are incomplete')
            import hashlib
            for e in elements:
                p = pages[e.pdf_page]
                if (e.paper_id != paper_id or e.document_sha256 != record['document_sha256']
                        or e.canonical_page_sha256 != hashlib.sha256(p['text'].encode()).hexdigest()):
                    raise ValueError('Formula source coordinate system mismatch')
                for s in e.source_spans+e.context+e.references:
                    source = pages[s.pdf_page]['text']
                    if (s.canonical_page_sha256 != hashlib.sha256(source.encode()).hexdigest()
                            or source[s.source_start:s.source_end] != s.text):
                        raise ValueError('Formula source span mismatch')
            self._loaded[paper_id] = {'record': record, 'manifest': manifest, 'elements': elements, 'pages': pages}
        return self._loaded[paper_id]

    def image(self, paper_id, element_id):
        document = self.document(paper_id)
        e = next((e for e in document['elements'] if e.element_id == element_id), None) if document else None
        if not e:
            raise KeyError('Formula is not part of the active document')
        path = (self.root/e.image_path).resolve()
        if not path.is_relative_to(self.root) or digest_file(path) != e.image_sha256:
            raise ValueError('Formula source image was changed')
        return path
