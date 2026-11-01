"""Inspect a frozen retrieved context, then generate one cited answer.

Default shows a prompt preview from the single-query routed development cache.
--generate calls the configured DeepSeek service once and caches a valid answer.
--heldout uses the completed final acceptance, including unsupported questions.
No retrieval, query decomposition or model-weight loading is performed.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_questions
from rag.generation import SYSTEM_PROMPT, generate, generation_settings, validate_answer
from rag.storage import atomic_json, read_json


def latest_development_cache():
    for directory in sorted((ROOT/'runtime/experiments').glob('decomposition-*'), reverse=True):
        state = read_json(directory/'run.json')
        if state and state['status'] == 'completed':
            return directory, state
    raise ValueError('找不到已完成的开发集检索缓存。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_options = parser.add_mutually_exclusive_group()
    source_options.add_argument('--source', type=Path, help='Completed decomposition cache directory')
    source_options.add_argument('--heldout', action='store_true', help='Read the completed frozen heldout acceptance')
    parser.add_argument('--question-id', help='Default is the first cached question')
    parser.add_argument('--generate', action='store_true', help='Allow actual DeepSeek answer generation')
    args = parser.parse_args()
    if args.heldout:
        frozen = read_json(ROOT/'runtime/acceptance/frozen_baseline.json')
        if not frozen:
            raise ValueError('请先运行最终验收。')
        source = ROOT/'runtime/acceptance'/('heldout-'+frozen['spec_sha256'][:12])
        state, split = read_json(source/'run.json'), 'heldout'
    else:
        source, state = (args.source, read_json(args.source/'run.json')) if args.source else latest_development_cache()
        split = 'dev'
    if not state or state['status'] != 'completed':
        raise ValueError('请选择已完成的检索结果目录。')
    qid = args.question_id or state['question_ids'][0]
    if qid not in state['question_ids']:
        parser.error('question-id must occur in the source cache')
    row_path = source/(f'{qid}.json' if args.heldout else f'{qid}_single_query.json')
    row = read_json(row_path)
    question = next(q for q in load_questions(split) if q['question_id'] == qid)
    if row['question'] != question['question']:
        raise ValueError('缓存问题与当前基准不匹配。')
    context = row['budget_context']
    settings = generation_settings()
    sources = [{'id': h['citation_id'], 'paper_id': h['paper_id'], 'title': h.get('title', ''),
                'pdf_page': h.get('pdf_page'), 'text': h['text']} for h in context]
    if len({h['id'] for h in sources}) != len(sources):
        raise ValueError('重复的证据引用编号。')
    request = {'model': settings.model, 'max_tokens': settings.max_tokens,
               'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                            {'role': 'user', 'content': json.dumps({'question': row['question'], 'evidence': sources}, ensure_ascii=False)}],
               'thinking': {'type': 'disabled'}, 'temperature': 0, 'response_format': {'type': 'json_object'}}
    identity = hashlib.sha256(json.dumps({'request': request, 'base_url': settings.base_url}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    target = ROOT/'runtime/generation_learning'/identity
    atomic_json(target/'prompt_preview.json', {'source_row': str(row_path), 'question_id': qid,
                                             'request_without_credentials': request,
                                             'source_row_sha256': hashlib.sha256(row_path.read_bytes()).hexdigest()})
    print(qid, row['question'])
    print('Evidence:', len(context), 'serialized chars:', row['context_chars'], 'model:', settings.model)
    for h in sources:
        print(h['id'], h['title'], 'PDF', h['pdf_page'])
    print('Prompt preview:', target/'prompt_preview.json')
    cached = read_json(target/'answer.json')
    if not args.generate and not cached:
        print('Add --generate to produce one answer using the configured API.')
        return
    if cached:
        checked = validate_answer(json.dumps({k: cached['answer'][k] for k in ['answer', 'citations', 'insufficient_evidence']}, ensure_ascii=False), context)
        answer = {**cached['answer'], **checked.model_dump()}
        print('Using cached answer.')
    else:
        if not context:
            print('No evidence; generation skipped. Insufficient evidence.')
            return
        answer = generate(row['question'], context, settings)
        atomic_json(target/'answer.json', {'question_id': qid, 'source_row': str(row_path), 'answer': answer,
                    'manual_review': {'claims_supported_by_citations': None, 'reference_facts_covered': None,
                                      'insufficiency_decision_correct': None},
                    'evaluation_only': {'expected_answerable': question['answerable'],
                                        'reference_answer': question.get('reference_answer', '')}})
    print('\nAnswer:', answer['answer'])
    print('Citations:', answer['citations'], 'insufficient_evidence:', answer['insufficient_evidence'])
    print('Usage:', answer.get('usage', {}))
    print('Reference points (evaluation only, never sent to generator):', question.get('reference_answer', ''))
    print('Citation validation checks IDs and consistency; factual support requires checking the cited text.')
    print('Saved answer:', target/'answer.json')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
