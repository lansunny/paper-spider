"""Gold-independent execution and source-anchor evaluation for A/B/C arms."""
from collections import defaultdict
from copy import deepcopy
from time import perf_counter

import numpy as np

from rag.benchmark import score_evidence
from rag.community_retrieval import merge_candidates,pack_context,source_key,subtract_interval
from rag.graph_evaluation import clustered_interval,covered_units,paper_clusters
from rag.retrieval import rerank_candidates

VARIANTS=('A','B','C')
STAGES=('candidate_pool','candidate40','top5','context_budget')


def original_candidates(pipeline,question,resolver,*,exact=True):
    start=perf_counter()
    result=pipeline.retrieve(question,mode='hybrid',top_k=5,rerank=False,collect_trace=True,exact=exact)
    # Keep the V2 paper-facet balancing order, then refill from its complete fused pool.
    prefix=result['rankings']['rrf20'];seen={h['chunk_id'] for h in prefix}
    raw=[*prefix,*[h for h in result['rankings']['union'] if h['chunk_id'] not in seen]]
    hits=[]
    for hit in raw:
        hits.append(resolver.canonical({**hit,'retrieval_paths':[{'branch':'fulltext','original_chunk_id':hit['chunk_id']}]}))
    return hits,result,round((perf_counter()-start)*1000,2)


def run_arm(question,fulltext,anchors,models,policy,*,rrf_k=60):
    """Only original question, original passages and fixed policy enter ranking."""
    started=perf_counter();merged,dedup=merge_candidates(fulltext,anchors,rrf_k=rrf_k)
    chosen=merged[:policy.rerank_limit]
    selection_ms=(perf_counter()-started)*1000
    started=perf_counter()
    ranked=rerank_candidates(models,question,chosen,title=True)
    rerank_ms=(perf_counter()-started)*1000
    if len(ranked)!=len(chosen) or {h['chunk_id'] for h in ranked}!={h['chunk_id'] for h in chosen}:
        raise ValueError('Unified reranker changed the candidate set')
    started=perf_counter();context,usage=pack_context(ranked,policy.context_chars)
    return {'rankings':{'candidate_pool':merged,'candidate40':chosen,'top5':ranked[:5],'context_budget':context},
            'reranked_candidates':ranked,'context_usage':usage,'deduplication':dedup,
            'rerank_pairs':len(chosen),'physical_rerank_pairs':len(chosen),
            'timings':{'merge_ms':round(selection_ms,2),'rerank_ms':round(rerank_ms,2),'context_ms':round((perf_counter()-started)*1000,2)}}


def score_arm(question,result,pages):
    """Labels are introduced here, after every candidate/ranking/context is fixed."""
    metrics={stage:score_evidence(question,hits,pages,k=max(1,len(hits))) for stage,hits in result['rankings'].items()}
    gold={s['paper_id'] for u in question['evidence_units'] for a in u['alternatives'] for s in a}
    for stage,metric in metrics.items():
        found={h['paper_id'] for h in result['rankings'][stage]}
        metric.update(paper_recall=len(gold&found)/len(gold),gold_papers_found=sorted(gold&found),gold_papers_missing=sorted(gold-found))
    return metrics


def novelty(anchors,fulltext):
    intervals=defaultdict(list)
    for h in fulltext:intervals[source_key(h)].extend((s['source_start'],s['source_end']) for s in h['source_spans'])
    new=0
    for h in anchors:
        for s in h['source_spans']:
            ranges=subtract_interval(s['source_start'],s['source_end'],intervals[source_key(h)])
            new+=sum(b-a for a,b in ranges);intervals[source_key(h)].extend(ranges)
    return new


def run_case(case,pipeline,retriever,resolver,models,policy,pages,order):
    question=case['question'];q=question['question']
    fulltext,raw,shared_ms=original_candidates(pipeline,q,resolver)
    route=raw['route'];single=route['mode']=='single_paper' or len(route.get('paper_ids',[]))==1
    # One embedding already exists in the full-text branch's encoder cache.
    vector=None if single else models.query_vector(q)
    branch_trace={'A':{'status':'fulltext_only','elapsed_ms':0}};branches={'A':[]}
    if not single:
        for variant in [v for v in order if v!='A']:
            anchors,trace=retriever.retrieve(q,variant,route,pipeline._router,resolver.manifest,vector)
            branches[variant]=[resolver.canonical(h) for h in anchors];branch_trace[variant]=trace
    results={}
    if single:
        baseline=run_arm(q,fulltext,[],models,policy,rrf_k=pipeline.settings.rrf_k)
        for variant in VARIANTS:
            results[variant]=deepcopy(baseline)
            if variant!='A':results[variant]['physical_rerank_pairs']=0
            branch_trace[variant]={'status':'protected_single_paper','elapsed_ms':0}
    else:
        for variant in order:results[variant]=run_arm(q,fulltext,branches[variant],models,policy,rrf_k=pipeline.settings.rrf_k)
    for variant,value in results.items():
        value['metrics']=score_arm(question,value,pages)
        value['branch']=branch_trace[variant]
        value['branch']['new_original_characters']=novelty(branches.get(variant,[]),fulltext)
        value['branch']['raw_anchor_source_ids']=[h['chunk_id'] for h in branches.get(variant,[])]
        value['branch']['anchor_evidence_units']=score_evidence(question,branches.get(variant,[]),pages,k=max(1,len(branches.get(variant,[]))))
        value['serial_stage_ms']=shared_ms+sum(value['timings'].values())+branch_trace[variant]['elapsed_ms']
        value['scope_guard_passed']=True
        if len(value['rankings']['candidate40'])>policy.rerank_limit or value['context_usage']['used_chars']>policy.context_chars:
            raise ValueError('Candidate or context budget exceeded')
        if any(h['paper_id'] not in resolver.manifest['documents'] for hits in value['rankings'].values() for h in hits):
            raise ValueError('Retrieved source escaped the experiment corpus')
        if variant!='A' and single and value['rankings']!=results['A']['rankings']:
            raise ValueError('A protected single-paper case changed')
        anchors_only=[h for h in value['rankings']['context_budget'] if all(p['branch']!='fulltext' for p in h['retrieval_paths'])]
        without=[h for h in value['rankings']['context_budget'] if h not in anchors_only]
        remaining=score_evidence(question,without,pages,k=max(1,len(without)))
        value['branch']['context_anchor_only_candidates']=len(anchors_only)
        value['branch']['units_requiring_anchor_only_context']=sorted(covered_units(value['metrics']['context_budget'])-covered_units(remaining))
    changes={}
    for newer,older in [('B','A'),('C','A'),('C','B')]:
        changes[newer+'_minus_'+older]={}
        for stage in STAGES:
            a=covered_units(results[newer]['metrics'][stage]);b=covered_units(results[older]['metrics'][stage])
            changes[newer+'_minus_'+older][stage]={'gained_units':sorted(a-b),'lost_units':sorted(b-a)}
    return {'question_id':question['question_id'],'question':q,'dataset':case['dataset'],
        'category':question['category'],'single_paper_guard':single,'route':route,'query_plan':raw['query_plan'],
        'shared_fulltext_ms':shared_ms,'variant_order':order,'results':results,'changes':changes,
        'guard_exact_match':single and all(results[v]['rankings']==results['A']['rankings'] for v in ('B','C'))}


def summarize(rows,cases,seed=42):
    clusters=paper_clusters(cases)
    scopes={'all':rows,'primary_non_single':[r for r in rows if not r['single_paper_guard']],
            'single_paper_guard':[r for r in rows if r['single_paper_guard']]}
    for dataset in sorted({r['dataset'] for r in rows}):scopes['dataset:'+dataset]=[r for r in rows if r['dataset']==dataset]
    for category in sorted({r['category'] for r in rows}):scopes['category:'+category]=[r for r in rows if r['category']==category]
    output={}
    for scope,records in scopes.items():
        if not records:continue
        variants={}
        for variant in VARIANTS:
            values=[r['results'][variant] for r in records]
            variants[variant]={'questions':len(records),'stages':{stage:{
                'evidence_coverage':float(np.mean([v['metrics'][stage]['evidence_unit_recall'] for v in values])),
                'complete_evidence_rate':float(np.mean([v['metrics'][stage]['all_required_evidence'] for v in values])),
                'paper_recall':float(np.mean([v['metrics'][stage]['paper_recall'] for v in values]))} for stage in STAGES},
                'median_serial_stage_ms':float(np.median([v['serial_stage_ms'] for v in values])),
                'mean_rerank_pairs':float(np.mean([v['rerank_pairs'] for v in values])),
                'mean_physical_rerank_pairs':float(np.mean([v['physical_rerank_pairs'] for v in values])),
                'mean_context_chars':float(np.mean([v['context_usage']['used_chars'] for v in values])),
                'new_original_characters':sum(v['branch']['new_original_characters'] for v in values),
                'anchor_dependent_context_units':sum(len(v['branch']['units_requiring_anchor_only_context']) for v in values)}
        comparisons={}
        for newer,older in [('B','A'),('C','A'),('C','B')]:
            deltas=[r['results'][newer]['metrics']['context_budget']['evidence_unit_recall']-r['results'][older]['metrics']['context_budget']['evidence_unit_recall'] for r in records]
            comparisons[newer+'_minus_'+older]={'mean_delta':float(np.mean(deltas)),
                'wins':sum(d>1e-12 for d in deltas),'losses':sum(d< -1e-12 for d in deltas),'ties':sum(abs(d)<=1e-12 for d in deltas),
                'uncertainty':clustered_interval(deltas,[clusters[r['question_id']] for r in records],seed=seed)}
        output[scope]={'variants':variants,'comparisons':comparisons}
    return output
