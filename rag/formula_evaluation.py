"""Independent original-PDF hard-case labels; no detector-defined recall denominator."""
import hashlib
import json

from rag.formula_schema import FormulaAnnotation
from rag.formula_store import FormulaStore


def intersection(a, b):
    return max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))


def evaluate_formula_regression(annotation_path, root):
    data=json.loads(annotation_path.read_text(encoding='utf-8'))
    labels=[FormulaAnnotation.model_validate(a) for a in data['annotations']]
    if len({a.annotation_id for a in labels}) != len(labels):
        raise ValueError('Duplicate annotation IDs')
    store=FormulaStore(root); results=[]
    for a in labels:
        d=store.document(a.paper_id,a.document_sha256)
        if not d:
            raise ValueError('No authoritative extraction matches annotation PDF')
        text=d['pages'][a.pdf_page]['text']
        if hashlib.sha256(text.encode()).hexdigest()!=a.canonical_page_sha256 or text[a.source_start:a.source_end]!=a.source_quote:
            raise ValueError('Annotation source coordinates changed; explicit realignment is required')
        def matches(e):
            if e.pdf_page!=a.pdf_page or (a.equation_number is not None and e.equation_number!=a.equation_number):
                return False
            # Ignore unrelated nearby formulas; compare against fixed source geometry.
            area=(a.bbox[2]-a.bbox[0])*(a.bbox[3]-a.bbox[1])
            return intersection(e.bbox,a.bbox)/area >= .25
        candidates=[e for e in d['elements'] if matches(e)]
        pending=[p for p in d['manifest']['pending_layout'] if p['pdf_page']==a.pdf_page
                 and p.get('equation_number')==a.equation_number and intersection(p['anchor_bbox'],a.bbox)>0]
        passed=bool(candidates) if a.label=='formula' else not (candidates or pending)
        results.append({'annotation_id':a.annotation_id,'group':a.group,'label':a.label,'passed':passed,
                        'matched_element_ids':[e.element_id for e in candidates],'pending':len(pending)})
    positive=[r for r in results if r['label']=='formula']; negative=[r for r in results if r['label']!='formula']
    return {'scope':'hard_case_regression_not_exhaustive_ground_truth',
            'annotation_version':data['version'], 'positive_retained':sum(r['passed'] for r in positive),
            'positive_cases':len(positive),'negative_rejected':sum(r['passed'] for r in negative),
            'negative_cases':len(negative),'all_passed':all(r['passed'] for r in results),'results':results,
            'note':'These selected hard cases do not establish corpus-wide precision or recall.'}
