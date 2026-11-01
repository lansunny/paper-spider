"""Formula retrieval labels refer to original PDF locations, never extraction IDs."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.chunking import normalized_blocks
from rag.formula_schema import Box, FormulaSourceText
from rag.formula_retrieval import aligned_spans
from rag.formula_store import FormulaStore
from rag.storage import atomic_json, atomic_text, read_json


class GoldFormula(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    gold_id: str
    paper_id: str
    title: str
    document_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    pdf_path: str
    source_url: str
    pdf_page: int = Field(ge=1)
    bbox: Box
    equation_number: str | None
    canonical_page_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_spans: list[FormulaSourceText] = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    image_path: str
    image_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    annotation_origin: Literal['agent_original_pdf_review'] = 'agent_original_pdf_review'
    human_verified: Literal[False] = False

    @model_validator(mode='after')
    def original_source_geometry(self):
        if (not all(math.isfinite(v) for v in self.bbox) or self.bbox[0]<0 or self.bbox[1]<0
                or self.bbox[2]<=self.bbox[0] or self.bbox[3]<=self.bbox[1]):
            raise ValueError('Gold formula must have a valid original PDF rectangle')
        if any(s.pdf_page!=self.pdf_page or s.canonical_page_sha256!=self.canonical_page_sha256 for s in self.source_spans):
            raise ValueError('Gold source spans must use the same page/coordinate system')
        return self


class FormulaQuestion(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    question_id: str
    question: str = Field(min_length=1)
    question_en: str = Field(min_length=1)
    category: Literal['semantic_single', 'semantic_open', 'semantic_cross', 'exact_number', 'literal_symbol', 'control']
    expectation: Literal['answerable', 'no_formula', 'no_exact_number', 'no_unscoped_exact'] = 'answerable'
    # Outer groups are AND; inner IDs are alternative acceptable source anchors.
    gold_groups: list[list[str]] = Field(default_factory=list)
    reference_answer: str = Field(min_length=1)
    requested_equation_number: str | None = None

    @model_validator(mode='after')
    def meaningful_groups(self):
        if (self.expectation == 'answerable') != bool(self.gold_groups):
            raise ValueError('Answerable questions require gold groups; controls must not have gold groups')
        if any(not group or len(group) != len(set(group)) for group in self.gold_groups):
            raise ValueError('Gold alternative groups must be nonempty and unique')
        if self.category == 'control' and self.expectation == 'answerable':
            raise ValueError('Control questions require a control expectation')
        if self.expectation == 'no_exact_number' and self.requested_equation_number is None:
            raise ValueError('Missing requested equation number for control')
        return self


def sha(value):
    return hashlib.sha256(value).hexdigest()


def native_formula_spans(page, box):
    """Map manually reviewed PDF rectangles to canonical source characters.

    No formula detector, extracted element ID, or retrieval result is used.
    """
    blocks = [{'block_number': b[5], 'text': b[4], 'bbox': tuple(b[:4])}
              for b in page.get_text('blocks', sort=False) if b[6] == 0]
    canonical, ranges = normalized_blocks(blocks)
    nonempty = [b for b in blocks if b['text'].strip()]
    mapping = {b['block_number']: (a, z) for b, (a, z, _) in zip(nonempty, ranges, strict=True)}
    import re
    cursor_by_block, sources = {}, []
    flags = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    target = pymupdf.Rect(box)
    for block in page.get_text('rawdict', flags=flags)['blocks']:
        if block['type'] != 0 or block['number'] not in mapping:
            continue
        block_start, block_end = mapping[block['number']]
        text = canonical[block_start:block_end]
        for line in block['lines']:
            chars = [c for s in line['spans'] for c in s['chars']]
            raw = ''.join(c['c'] for c in chars)
            normalized = re.sub(r'\s+', ' ', raw).strip()
            at = text.find(normalized, cursor_by_block.get(block['number'], 0))
            if not normalized or at < 0:
                continue
            cursor_by_block[block['number']] = at+len(normalized)
            selected, cursor = [], 0
            for c in chars:
                r = pymupdf.Rect(c['bbox'])
                center = pymupdf.Point((r.x0+r.x1)/2, (r.y0+r.y1)/2)
                if c['c'].strip() and target.contains(center):
                    selected.append((cursor, cursor+len(c['c']), c['bbox']))
                cursor += len(c['c'])
            if not selected:
                continue
            a = block_start+at+len(re.sub(r'\s+', ' ', raw[:selected[0][0]]).lstrip())
            z = block_start+at+len(re.sub(r'\s+', ' ', raw[:selected[-1][1]]).lstrip())
            if a >= z:
                continue
            rect = pymupdf.Rect(selected[0][2])
            for _, _, b in selected[1:]:
                rect |= pymupdf.Rect(b)
            sources.append(FormulaSourceText(pdf_page=page.number+1, block_number=block['number'], bbox=tuple(rect),
                text=canonical[a:z], source_start=a, source_end=z, canonical_page_sha256=sha(canonical.encode()),
                relation='formula_body'))
    return canonical, sources


def prepare_dataset(directory, root, force=False):
    """Curation rectangles/hashes are fixed; extraction changes cannot re-label gold."""
    curation = read_json(directory/'curation.json')
    if not curation:
        raise ValueError('Missing manually curated formula questions and source locations')
    if (directory/'manifest.json').exists() and not force:
        return load_dataset(directory, root, validate_sources=True)[2]
    gold = []
    for item in curation['gold_sources']:
        source = (root/item['pdf_path']).resolve()
        if not source.is_relative_to(root.resolve()) or sha(source.read_bytes()) != item['document_sha256']:
            raise ValueError('Curated original PDF changed: '+item['paper_id'])
        with pymupdf.open(source) as pdf:
            page = pdf[item['pdf_page']-1]
            page.set_rotation(0)
            canonical, spans = native_formula_spans(page, item['bbox'])
            path = directory/'gold_images'/(item['gold_id']+'.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            bbox = pymupdf.Rect(item['bbox'])
            crop = pymupdf.Rect(bbox.x0-3,bbox.y0-3,bbox.x1+3,bbox.y1+3)&page.rect
            page.get_pixmap(matrix=pymupdf.Matrix(2.5,2.5), clip=crop, alpha=False, annots=False).save(path)
        gold.append(GoldFormula(**item, source_spans=spans, canonical_page_sha256=sha(canonical.encode()),
                               image_path=path.relative_to(root).as_posix(), image_sha256=sha(path.read_bytes())))
    questions = [FormulaQuestion.model_validate(q) for q in curation['questions']]
    atomic_text(directory/'gold_formulas.jsonl', ''.join(g.model_dump_json()+'\n' for g in gold))
    atomic_text(directory/'questions.jsonl', ''.join(q.model_dump_json()+'\n' for q in questions))
    manifest = {'version': curation['version'], 'questions': len(questions), 'gold_formulas': len(gold),
                'source_papers': len({g.paper_id for g in gold}),
                'categories': dict(Counter(q.category for q in questions)),
                'gold_sha256': sha((directory/'gold_formulas.jsonl').read_bytes()),
                'questions_sha256': sha((directory/'questions.jsonl').read_bytes()),
                'curation_sha256': sha((directory/'curation.json').read_bytes()),
                'human_verified': False, 'answer_generation_evaluated': False,
                'selection': 'Source-reviewed subset of detected candidates; does not measure detector recall.',
                'limitations': ['Agent-reviewed annotations, not independent human labels.',
                    'Open-query gold is a non-exhaustive set of source anchors; report anchor recall, not precision.',
                    'Several question forms share the same gold formula; 50 questions are not 50 independent formulas.',
                    'Report 35 semantic questions separately from 11 lookup diagnostics and 4 controls.',
                    'Manual faithful English translations are supplied to BM25; this is not an online translation system.']}
    atomic_json(directory/'manifest.json', manifest)
    load_dataset(directory, root, validate_sources=True)
    return manifest


def load_dataset(directory, root, validate_sources=True):
    meta = read_json(directory/'manifest.json')
    if not meta:
        raise ValueError('Prepare the formula test50 dataset first')
    for name, key in [('gold_formulas.jsonl','gold_sha256'),('questions.jsonl','questions_sha256'),('curation.json','curation_sha256')]:
        if sha((directory/name).read_bytes()) != meta[key]:
            raise ValueError('Formula dataset integrity mismatch: '+name)
    gold_rows = [GoldFormula.model_validate_json(s) for s in (directory/'gold_formulas.jsonl').read_text(encoding='utf-8').splitlines() if s]
    questions = [FormulaQuestion.model_validate_json(s) for s in (directory/'questions.jsonl').read_text(encoding='utf-8').splitlines() if s]
    gold = {g.gold_id:g for g in gold_rows}
    if len(gold) != len(gold_rows) or len({q.question_id for q in questions}) != len(questions):
        raise ValueError('Duplicate formula/question annotation IDs')
    if len(questions) != meta['questions'] or len(gold) != meta['gold_formulas']:
        raise ValueError('Formula dataset size mismatch')
    used = {gid for q in questions for group in q.gold_groups for gid in group}
    if used != set(gold):
        raise ValueError('Unknown or unused gold source annotations')
    if validate_sources:
        by_source = {}
        for g in gold.values():
            source = (root/g.pdf_path).resolve()
            if source not in by_source:
                if not source.is_relative_to(root.resolve()):
                    raise ValueError('Gold source outside project')
                by_source[source] = sha(source.read_bytes())
            if by_source[source] != g.document_sha256:
                raise ValueError('Gold PDF changed: '+g.paper_id)
            with pymupdf.open(source) as pdf:
                p=pdf[g.pdf_page-1]; p.set_rotation(0)
                canonical,_=native_formula_spans(p,g.bbox)
                if sha(canonical.encode()) != g.canonical_page_sha256:
                    raise ValueError('Gold canonical source coordinates changed')
                for span in g.source_spans:
                    if span.pdf_page != g.pdf_page or canonical[span.source_start:span.source_end] != span.text:
                        raise ValueError('Gold source quote mismatch')
            image = (root/g.image_path).resolve()
            if not image.is_relative_to(root.resolve()) or sha(image.read_bytes()) != g.image_sha256:
                raise ValueError('Gold original-PDF crop changed')
    return questions, gold, meta


def formula_matches(formula, gold, min_coverage=.7):
    if (formula['paper_id'] != gold.paper_id or formula['document_sha256'] != gold.document_sha256
            or formula['pdf_page'] != gold.pdf_page):
        return False
    if gold.equation_number is not None and formula.get('equation_number') != gold.equation_number:
        return False
    box, target = pymupdf.Rect(formula['bbox']), pymupdf.Rect(gold.bbox)
    overlap = box & target
    return not overlap.is_empty and overlap.width*overlap.height/(target.width*target.height) >= min_coverage


def score_formulas(question, formulas, gold, *, joins_only=False):
    join_methods = {'formula_inside_chunk','adjacent_explanation','explicit_formula_reference'}
    eligible = [f for f in formulas if not joins_only or any(m['method'] in join_methods for m in f['matches'])]
    if question.expectation != 'answerable':
        if question.expectation == 'no_formula':
            passed = not eligible
        elif question.expectation == 'no_exact_number':
            passed = not any(f.get('equation_number') == question.requested_equation_number for f in eligible)
        else:
            passed = not any(m['method']=='exact_equation_number' for f in eligible for m in f['matches'])
        return {'anchor_recall': None,'all_required': None,'control_passed': passed,
                'returned_formulas':len(eligible),'note':'Control outcome is not answer-generation accuracy.'}
    groups=[]
    for group in question.gold_groups:
        hits=[(i+1,gid,f['element_id']) for i,f in enumerate(eligible) for gid in group if formula_matches(f,gold[gid])]
        groups.append({'alternatives':group,'covered':bool(hits), 'matches':hits, 'first_formula_rank':min((h[0] for h in hits),default=None)})
    return {'anchor_recall':sum(g['covered'] for g in groups)/len(groups),
            'all_required':all(g['covered'] for g in groups),'groups':groups,'returned_formulas':len(eligible)}


def source_anchor_hits(question, hits, gold, store):
    """Auxiliary source overlap; no gold filters are sent to the retriever."""
    intervals={}
    for hit in hits:
        key=(hit['paper_id'],hit['document_sha256'],hit.get('pdf_page'))
        if key not in intervals:
            intervals[key]=[]
        d=store.document(key[0],key[1])
        if d and key[2] in d['pages']:
            spans,_=aligned_spans(hit,d['pages'][key[2]])
            intervals[key].extend(spans)
    covered=[]
    for group in question.gold_groups:
        found=False
        for gid in group:
            g=gold[gid]; spans=intervals.get((g.paper_id,g.document_sha256,g.pdf_page),[])
            if any(a < s.source_end and s.source_start < z for s in g.source_spans for a,z in spans):
                found=True
        covered.append(found)
    return {'groups':covered,'anchor_overlap_rate':sum(covered)/len(covered) if covered else None,
            'definition':'At least one original formula source character overlaps a retrieved chunk; not complete evidence coverage.'}
