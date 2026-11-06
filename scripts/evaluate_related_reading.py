"""Run a frozen small diagnostic of reading recommendations; no graph/index changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from rag.embeddings import LocalModels
from rag.formula_store import digest_file
from rag.generation import generation_settings
from rag.ingestion import fingerprint
from rag.pipeline_v2 import load_v2_settings
from rag.reading_evaluation import (VERSION,TASKS,RELEVANCE_SYSTEM,AUDIT_SYSTEM,OriginalPackets,
    direct_candidates,navigation_payload,select_navigation,relevance_payload,validate_relevance,
    audit_payload,validate_audit,list_metrics,summarize)
from rag.related_reading import graph_candidates,PLAN_SYSTEM
from rag.reviews.cards import load_cards,request_step
from rag.reviews.corpus import ReviewCorpus
from rag.storage import atomic_json,read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--judge-model',default='deepseek-v4-pro')
    parser.add_argument('--instrumentation-amendment',help='Document a code-only instrumentation repair; tasks, prompts and rubric must remain identical')
    args=parser.parse_args()
    corpus=ReviewCorpus(ROOT);card_manifest,cards=load_cards(ROOT,corpus)
    tasks=TASKS
    previous=read_json(ROOT/'runtime/related-reading-acceptance.json')
    result_path=ROOT/'runtime/related_reading/jobs'/previous['run_id']/'result.json'
    current=read_json(result_path)
    if current['graph_build_id']!=corpus.build_id:raise ValueError('The audited example belongs to an older graph')
    for hit in current['evidence']:
        page=corpus.canonical_pages(hit['paper_id'])[hit['pdf_page']]
        if hit['document_sha256']!=corpus.papers[hit['paper_id']]['document_sha256']:
            raise ValueError('Audited evidence PDF version changed')
        for span in hit['source_spans']:
            if hit['text'][span['text_start']:span['text_end']]!=page['text'][span['source_start']:span['source_end']]:
                raise ValueError('Audited evidence no longer matches original source offsets')
    paths=['rag/reading_evaluation.py','scripts/evaluate_related_reading.py','rag/related_reading.py',
           'rag/reviews/cards.py','rag/paper_graph/extraction.py']
    design={'version':VERSION,'graph_build_id':corpus.build_id,'papers':len(corpus.papers),
        'cardset_id':card_manifest['cardset_id'],'tasks':tasks,'selection_model':'deepseek-flash','judge_model':args.judge_model,
        'pool_size':12,'recommendation_limit':5,'selection_max_tokens':1600,'original_text_chars_per_paper':3000,
        'A':'dimension-vector pool + goal-specific selection, no community cards',
        'B':'graph pool + same selection, no community cards',
        'C':'same graph pool + same selection + community cards',
        'equalized_input_fields':'paper ID, title, four descriptions; no scores/directions/memberships in any arm',
        'judge':'blinded pooled original-source review, shuffled papers, no arm/rank/graph summaries',
        'source_audit':'the existing production example only, not generated again',
        'metrics':'useful (grade>=2) count / fixed 5 slots; returned precision separately; pooled pre-evidence nDCG',
        'no_online_body_retrieval_or_answer_generation':True,
        'code_hashes':{p:digest_file(ROOT/p) for p in paths},'audited_result_sha256':digest_file(result_path),
        'prompts':{'selection':fingerprint(PLAN_SYSTEM),'relevance':fingerprint(RELEVANCE_SYSTEM),'audit':fingerprint(AUDIT_SYSTEM)}}
    key=fingerprint(design)
    directory=args.resume or ROOT/'runtime/experiments'/('reading-quality-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    if args.resume:
        old=read_json(directory/'run.json')
        if old['design_sha256']!=key:
            same=lambda d:{k:v for k,v in d.items() if k!='code_hashes'}
            if not args.instrumentation_amendment or same(old['design'])!=same(design):
                raise ValueError('Frozen design changed; do not resume this run')
            amendments=old.get('instrumentation_amendments',[])+[{'reason':args.instrumentation_amendment,
                'previous_code_hashes':old['design']['code_hashes'],'new_code_hashes':design['code_hashes'],
                'unchanged':'tasks, selection outputs, prompts, semantic rubric, source windows and budgets'}]
        else:amendments=old.get('instrumentation_amendments',[])
    else:amendments=[]
    run={'design_sha256':key,'design':design,'status':'prepared'}
    if amendments:run['instrumentation_amendments']=amendments
    atomic_json(directory/'run.json',run)
    print('Frozen design:',directory,flush=True)
    if not args.run:return 0
    settings=generation_settings(ROOT).model_copy(update={'model':'deepseek-flash','timeout_seconds':180})
    judge=settings.model_copy(update={'model':args.judge_model,'max_tokens':6500})
    if not settings.configured:raise RuntimeError('A local API credential is required')
    run['status']='running';atomic_json(directory/'run.json',run)
    models=LocalModels(load_v2_settings());prepared={}
    for task in tasks:
        path=directory/'tasks'/task['task_id']/'pools.json'
        pools=read_json(path)
        if not pools:
            query=models.query_vector(task['goal'])
            pools={'direct':direct_candidates(corpus,task['paper_id'],query,12),
                'graph':graph_candidates(corpus,task['paper_id'],task['goal'],models,12)}
            atomic_json(path,pools)
        prepared[task['task_id']]=pools
    # Run only two bounded API requests at a time; every arm shares the same prompt and limits.
    def select(task,arm):
        path=directory/'tasks'/task['task_id']/(arm+'.json')
        old=read_json(path)
        if old:return task['task_id'],arm,old
        pool=prepared[task['task_id']]['direct' if arm=='A' else 'graph']
        payload=navigation_payload(corpus,task,pool,cards if arm=='C' else [])
        identity=fingerprint({'system':PLAN_SYSTEM,'payload':payload,'model':settings.model,
                              'endpoint':settings.base_url,'max_tokens':1600})
        failure_path=ROOT/'runtime/reviews/cache/model_steps'/(identity+'.failed.json')
        previous_attempts=len(read_json(failure_path,{}).get('attempts',[]))
        try:
            value=select_navigation(ROOT,corpus,task,pool,cards if arm=='C' else [],settings)
            value['status']='completed'
        except (ValueError,RuntimeError) as error:
            failures=read_json(failure_path,{}).get('attempts',[])[previous_attempts:]
            value={'status':'failed','error':str(error),'selected_paper_ids':[],
                'failed_cost':{'api_attempts':len(failures),'usage_current_request':{
                    k:sum(a.get('usage',{}).get(k,0) for a in failures) for k in ('prompt_tokens','completion_tokens','total_tokens')}}}
        atomic_json(path,value)
        return task['task_id'],arm,value
    selections={t['task_id']:{} for t in tasks}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(select,t,a) for i,t in enumerate(tasks) for a in ('ABC'[i%3:]+'ABC'[:i%3])]
        for f in as_completed(futures):
            task_id,arm,value=f.result();selections[task_id][arm]=value
            print('Selection:',task_id,arm,value['status'],len(value['selected_paper_ids']),flush=True)
    packets=OriginalPackets(corpus);cases=[];judge_steps=[]
    for i,task in enumerate(tasks):
        base=directory/'tasks'/task['task_id'];arms=selections[task['task_id']]
        found={p for a in arms.values() for p in a['selected_paper_ids']}
        if task['task_id']=='eval_metrics':found.update(r['paper_id'] for r in current['recommendations'])
        if not found:
            reviewed={'ratings':[]};aliases={};step=None
        else:
            payload,aliases=relevance_payload(corpus,packets,task,found,seed=42+i)
            atomic_json(base/'blinded-review-input.json',payload)
            reviewed,step=request_step(RELEVANCE_SYSTEM,payload,judge,ROOT,
                lambda v,p=payload:validate_relevance(v,p),'reading_quality_original_source_review')
            judge_steps.append(step)
        decoded=[{**r,'paper_id':aliases[r['paper']]} for r in reviewed['ratings']]
        grades={r['paper_id']:r['grade'] for r in decoded}
        case={'task':task,'arms':arms,'judgments':decoded,'judge_step':step,
            'metrics':{a:list_metrics(v['selected_paper_ids'],grades) for a,v in arms.items()}}
        if task['task_id']=='eval_metrics':case['existing_production_metrics']=list_metrics(current['reading_order'],grades)
        atomic_json(base/'evaluation.json',case);cases.append(case)
        print('Reviewed:',task['task_id'],{a:m['useful_count'] for a,m in case['metrics'].items()},flush=True)
    payload=audit_payload(current);atomic_json(directory/'existing-advice-audit-input.json',payload)
    audit,step=request_step(AUDIT_SYSTEM,payload,judge,ROOT,lambda v:validate_audit(v,payload),'reading_quality_citation_support_review')
    judge_steps.append(step);atomic_json(directory/'existing-advice-audit.json',audit)
    summary=summarize(cases,audit)
    steps=[v['step'] for arms in selections.values() for v in arms.values() if 'step' in v]+judge_steps
    steps += [v['failed_cost'] for arms in selections.values() for v in arms.values() if 'failed_cost' in v]
    summary['model_costs']={'recorded_successful_steps':len(steps),'api_attempts_current_request':sum(s.get('api_attempts',0) for s in steps),
        'usage_current_request':{k:sum(s.get('usage_current_request',{}).get(k,0) for s in steps) for k in ('prompt_tokens','completion_tokens','total_tokens')},
        'includes_failed_selection_response_usage':True,
        'usage_scope':'Successful API responses including schema-repair responses; network/server errors without usage are not token-counted'}
    summary['model_costs']['all_recorded_response_usage']={k:sum(s.get('usage',s.get('usage_current_request',{})).get(k,0) for s in steps)
        for k in ('prompt_tokens','completion_tokens','total_tokens')}
    summary['original_example']={'run_id':previous['run_id'],'source_ranges_verified':len(current['evidence']),
        'recommendations':len(current['recommendations']),'support_status_is_not_entailment':True}
    atomic_json(directory/'summary.json',summary)
    run['status']='completed';atomic_json(directory/'run.json',run)
    print('Result:',directory/'summary.json',flush=True)
    return 0


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
