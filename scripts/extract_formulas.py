"""Extract formula sidecars/crops with source provenance, no model/API calls.

PyCharm example: --paper-id 2024.eacl-demo.16 --pages 3
Without --pages, inspect the full selected paper (default: RAGAs).
External OCR import: --formula-directory <output> --import-ocr <results.jsonl>
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.formula_extraction import extract_document, import_ocr_results
from rag.ingestion import paper_catalog
from rag.formula_store import publish_document, FormulaStore


def project_path(path):
    return (path if path.is_absolute() else ROOT/path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-id', nargs='+', default=['2024.eacl-demo.16'])
    parser.add_argument('--pages', nargs='+', type=int, help='1-based PDF pages; default all')
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--formula-directory', type=Path)
    parser.add_argument('--import-ocr', type=Path)
    parser.add_argument('--status', action='store_true', help='Show authoritative full-document artifacts')
    args = parser.parse_args()
    if args.status:
        for pid, record in FormulaStore(ROOT).records.items():
            print(pid, record['extractor_version'], record['elements'], record['directory'])
        return
    if args.import_ocr:
        if not args.formula_directory:
            parser.error('--import-ocr requires --formula-directory')
        directory = project_path(args.formula_directory)
        if not directory.is_relative_to((ROOT/'runtime/formulas').resolve()):
            parser.error('formula-directory must be under runtime/formulas')
        results = [json.loads(s) for s in project_path(args.import_ocr).read_text(encoding='utf-8').splitlines() if s]
        count = import_ocr_results(directory, results, ROOT)
        print(f'Imported {count} OCR results as recognized_unverified; raw detections unchanged.')
        return
    if args.formula_directory:
        parser.error('--formula-directory is only used with --import-ocr')
    if args.pages and any(p < 1 for p in args.pages):
        parser.error('PDF pages start at 1')
    catalog = {p['paper_id']: p for p in paper_catalog()}
    for pid in dict.fromkeys(args.paper_id):
        if pid not in catalog:
            parser.error(f'Unknown downloaded paper: {pid}')
        paper = catalog[pid]
        path = (ROOT/paper['pdf_path']).resolve()
        output, elements, manifest = extract_document({**paper, '_path': path}, root=ROOT, pages=args.pages, dpi=args.dpi)
        print(f"{pid}: {len(elements)} candidates, {manifest['numbered']} numbered; cache={manifest['cache_hit']}", flush=True)
        for e in elements:
            print(f"  PDF {e.pdf_page} | Eq {e.equation_number or '-'} | {e.layout_kind} | {e.confidence_band} | {e.element_id[:8]}")
        print('Pending numbered-layout proposals:', len(manifest['pending_layout']))
        print('Pages needing OCR layout detection:', manifest['pages_needing_ocr_layout'])
        print('Metadata:', output/'elements.jsonl')
        print('OCR jobs:', output/'ocr_jobs.jsonl')
        if args.pages is None:
            publish_document(output, ROOT)
            print('Published full document to:', ROOT/'runtime/formulas/CURRENT.json')
        else:
            print('Partial run: active full-document artifact was not replaced.')
        print('All candidates remain unreviewed. No LaTeX was guessed or verified.')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
