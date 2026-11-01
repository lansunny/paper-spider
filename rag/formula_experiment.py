"""Stage-wise text retrieval and formula attachment evaluation; no answer generation."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import portalocker

from rag.formula_benchmark import formula_matches, score_formulas, source_anchor_hits, sha
from rag.formula_retrieval import attach_formulas
from rag.formula_store import FormulaStore
from rag.generation import select_context
from rag.ingestion import current_manifest, index_directory, fingerprint
from rag.storage import atomic_json, read_json

STAGES = ['bm25','dense','union','rrf20_before_formula_rescue','rrf20','after_top5','context_budget']


def preflight(settings, gold, root, store):
    manifest = current_manifest(settings,root)
    if not manifest or manifest['profile'] != settings.index_profile():
        raise ValueError('No complete text index matches the requested configuration')
    available={}
    for gid,g in gold.items():
        if manifest['documents'].get(g.paper_id) != g.document_sha256:
            raise ValueError('Gold PDF does not match the indexed source: '+g.paper_id)
        d=store.document(g.paper_id,g.document_sha256)
        matches=[e.element_id for e in d['elements'] if formula_matches(e.model_dump(),g)] if d else []
        available[gid]={'available':bool(matches),'current_element_ids':matches}
    return manifest,available


def compact_formula(f,store,check_image=False):
    value={k:v for k,v in f.items() if k not in {'text_proxy','image_url'}}
    d=store.document(f['paper_id'],f['document_sha256'])
    e=next(e for e in d['elements'] if e.element_id==f['element_id'])
    path=store.image(e.paper_id,e.element_id) if check_image else store.root/e.image_path
    value['image_path']=path.as_posix()
    return value


def diagnose(question,metrics,source,available,route,gold):
    if question.expectation!='answerable':
        return [] if metrics['context_budget']['control_passed'] else ['control_failed']
    missing=[]
    for position,group in enumerate(question.gold_groups):
        if metrics['context_budget']['groups'][position]['covered']:
            continue
        if not any(available[g]['available'] for g in group): reason='gold_formula_not_extracted'
        elif route.get('paper_ids') and not any(gold[g].paper_id in route['paper_ids'] for g in group): reason='routing_scope_excluded_gold'
        elif not metrics['union']['groups'][position]['covered']:
            reason=('formula_source_recalled_but_not_linked' if source['union']['groups'][position]
                    else 'no_gold_formula_link_from_recall_candidates')
        elif not metrics['rrf20']['groups'][position]['covered']: reason='formula_link_lost_at_candidate_trim'
        elif source['context_budget']['groups'][position]: reason='formula_attachment_or_display_limit_miss'
        elif metrics['after_top5']['groups'][position]['covered']: reason='context_budget_lost_formula_link'
        else: reason='rerank_or_context_budget_lost_formula_link'
        missing.append({'gold_group':group,'reason':reason})
    return missing


def evaluate_question(question,gold,pipeline,store,manifest,available,*,mode='hybrid',rerank=True,
                      top_k=5,formula_limit=12,english_lexical=True):
    start=perf_counter()
    # Only question text reaches retrieval. Gold IDs, pages, formulas and answers
    # are used strictly AFTER retrieval for scoring, not paper filters or prompts.
    result=pipeline.retrieve(question.question,mode=mode,top_k=top_k,rerank=rerank,
                             lexical_query=question.question_en if english_lexical else None,
                             auto_route=True,collect_trace=True)
    if result['snapshot_id'] != manifest['snapshot_id']:
        raise ValueError('Text index changed during the formula experiment')
    context,chars=select_context(result['reranked_candidates'],pipeline.settings.context_char_budget)
    candidates={s:result['rankings'][s] for s in STAGES if s in result['rankings']}
    candidates['context_budget']=context
    scope=result['route'].get('paper_ids') or None
    v2 = hasattr(pipeline, 'attach_formulas')
    facets = result.get('query_plan', {}).get('facets', [])
    metrics,join_metrics,sources,attachments={}, {}, {}, {}
    for stage,hits in candidates.items():
        # Diagnostic candidate stages do not discard matches with the UI limit.
        limit=formula_limit if stage in {'after_top5','context_budget'} else 100000
        if v2:
            from rag.formula_retrieval_v2 import attach_formulas_v2
            # Huge candidate-stage diagnostics only measure availability. Local
            # formula ranking is applied to the two actual presentation stages.
            attached=attach_formulas_v2(question.question,hits,paper_ids=scope,documents=manifest['documents'],
                root=pipeline.root,store=store,limit=limit,models=pipeline.models,facets=facets,
                rank=pipeline.policy.formula_ranking and stage in {'after_top5','context_budget'})
        else:
            attached=attach_formulas(question.question,hits,paper_ids=scope,documents=manifest['documents'],
                                     root=pipeline.root,store=store,limit=limit)
        metrics[stage]=score_formulas(question,attached['formulas'],gold)
        join_metrics[stage]=score_formulas(question,attached['formulas'],gold,joins_only=True)
        sources[stage]=source_anchor_hits(question,hits,gold,store)
        attachments[stage]=attached
    # Explicit lookup alone is a separately reported diagnostic, never the main score.
    direct=attach_formulas(question.question,[],paper_ids=scope,documents=manifest['documents'],
                           root=pipeline.root,store=store,limit=formula_limit)
    final=attachments['context_budget']
    return {'question_id':question.question_id,'question':question.question,'question_en':question.question_en,
            'category':question.category,'expectation':question.expectation,'question_sha256':fingerprint(question.model_dump()),
            'metrics':metrics,'join_only_metrics':join_metrics,'source_overlap':sources,
            'direct_lookup_only':score_formulas(question,direct['formulas'],gold),
            'route':result['route'],'candidate_counts':result['candidate_counts'],'context_chars':chars,
            'pipeline_version':result.get('pipeline_version','v1'), 'query_plan':result.get('query_plan'),
            'bm25_query':result.get('bm25_query'), 'bm25_query_source':result.get('bm25_query_source'),
            'candidate_selection':result.get('candidate_selection'),
            'context_budget':pipeline.settings.context_char_budget,'context_items':len(context),
            'context_candidate_items':len(result['reranked_candidates']),
            'formula_prefix_metrics':{str(k):score_formulas(question,final['formulas'][:k],gold)
                                      for k in [1,2,3,5,12]},
            'stage_chunk_ids':{s:[h['chunk_id'] for h in hits] for s,hits in candidates.items()},
            'stage_formula_ids':{s:[f['element_id'] for f in a['formulas']] for s,a in attachments.items()},
            'final_formulas':[compact_formula(f,store,check_image=True) for f in final['formulas']],
            'final_attachment_diagnostics':final['formula_retrieval'],
            'failure_diagnosis':diagnose(question,metrics,sources,available,result['route'],gold),
            'timings':result['timings'],'elapsed_ms':round((perf_counter()-start)*1000,2),
            'reference_answer':question.reference_answer,'reference_is_evaluation_only':True}


def summarize(rows):
    def group(selected):
        answerable=[r for r in selected if r['expectation']=='answerable']
        controls=[r for r in selected if r['expectation']!='answerable']
        stages={}
        for stage in STAGES:
            cohort=[r for r in answerable if stage in r['metrics']]
            if cohort:
                stages[stage]={'questions':len(cohort),
                    'macro_formula_anchor_recall':sum(r['metrics'][stage]['anchor_recall'] for r in cohort)/len(cohort),
                    'all_required_question_rate':sum(r['metrics'][stage]['all_required'] for r in cohort)/len(cohort),
                    'join_only_anchor_recall':sum(r['join_only_metrics'][stage]['anchor_recall'] for r in cohort)/len(cohort),
                    'mean_returned_formulas':sum(r['metrics'][stage]['returned_formulas'] for r in cohort)/len(cohort)}
        return {'questions':len(selected),'answerable':len(answerable),'controls':len(controls),
                'stages':stages,'controls_passed':sum(r['metrics']['context_budget']['control_passed'] for r in controls),
                'display_prefix':{str(k):{'macro_formula_anchor_recall':sum(
                    r['formula_prefix_metrics'][str(k)]['anchor_recall'] for r in answerable)/len(answerable),
                    'all_required_question_rate':sum(r['formula_prefix_metrics'][str(k)]['all_required']
                    for r in answerable)/len(answerable)} for k in [1,2,3,5,12]} if answerable else {}}
    semantic=[r for r in rows if r['category'].startswith('semantic_')]
    lookup=[r for r in rows if r['category'] in {'exact_number','literal_symbol'}]
    return {'semantic_main':group(semantic),'lookup_diagnostics':group(lookup),
            'all_answerable':group([r for r in rows if r['expectation']=='answerable']),
            'by_category':{c:group([r for r in rows if r['category']==c]) for c in sorted({r['category'] for r in rows})},
            'failures':{r['question_id']:r['failure_diagnosis'] for r in rows if r['failure_diagnosis']},
            'definitions':{'macro_formula_anchor_recall':'Per question: covered required formula groups / all required groups; alternatives within a group use OR; average across questions.',
                'all_required_question_rate':'Fraction of questions whose required formula groups are all covered, including every paper in cross-paper questions.',
                'join_only_anchor_recall':'Recall using text-source joins only, excluding number/symbol-only matches.',
                'negative_controls':'No-formula tests check unwanted attachments; number/ambiguity controls check lookup safety. These do not measure model refusal.',
                'rrf20':'V1: global RRF trim; V2: fixed candidate slots reserved per facet, optionally with source-linked formula rescue. Exact candidate count is recorded per question.',
                'no_qa_accuracy':'No answer generation or formula understanding is evaluated.'}}


def run_experiment(questions,gold,metadata,settings,root,*,mode='hybrid',rerank=True,top_k=5,
                   formula_limit=12,english_lexical=None,pipeline_factory=None,emit=None,
                   retrieval_profile='v1',policy=None):
    if retrieval_profile not in {'v1','v2'}:
        raise ValueError('Unknown retrieval profile')
    if english_lexical is None:
        english_lexical = retrieval_profile == 'v1'
    store=FormulaStore(root)
    manifest,available=preflight(settings,gold,root,store)
    code_paths=['rag/pipeline.py','rag/retrieval.py','rag/routing.py','rag/embeddings.py','rag/vector_store.py',
                'rag/generation.py','rag/formula_retrieval.py','rag/formula_store.py',
                'rag/formula_benchmark.py','rag/formula_experiment.py','configs/paper_aliases.json']
    if retrieval_profile == 'v2':
        from dataclasses import asdict
        from rag.pipeline_v2 import RetrievalPolicyV2
        policy = policy or RetrievalPolicyV2()
        code_paths += ['rag/pipeline_v2.py','rag/query_planning_v2.py','rag/query_decomposition.py',
                       'rag/facet_reranking.py','rag/lexical_v2.py','rag/formula_retrieval_v2.py',
                       'configs/lexical_terms_zh_en.json']
    profile={'dataset_questions_sha256':metadata['questions_sha256'],'dataset_gold_sha256':metadata['gold_sha256'],
             'formula_registry_sha256':sha((root/'runtime/formulas/CURRENT.json').read_bytes()),
             'snapshot_id':manifest['snapshot_id'],
             'snapshot_sha256':sha((index_directory(settings,root)/(manifest['snapshot_id']+'.jsonl')).read_bytes()),
             'settings':settings.model_dump(),'mode':mode,'rerank':rerank,'top_k':top_k,'formula_limit':formula_limit,
             'english_lexical_translation':english_lexical,'generation_api':False,
             'code_sha256':{p:sha((root/p).read_bytes()) for p in code_paths}}
    profile.update(retrieval_profile=retrieval_profile,
                   policy=asdict(policy) if retrieval_profile=='v2' else None,
                   candidate_trim_policy='per_paper_slots_plus_linked_formula_sources' if retrieval_profile=='v2' else 'global_rrf',
                   context_eligible='all_reranked_candidates')
    digest=fingerprint(profile)
    target=root/'runtime/experiments'/('formula-test50-'+digest[:16]);target.mkdir(parents=True,exist_ok=True)
    pipeline=None;rows=[]
    with portalocker.Lock(str(target/'run.lock'),timeout=0):
        state={'status':'running','profile':profile,'profile_sha256':digest,'selected_question_ids':[q.question_id for q in questions],
               'started_at':datetime.now(timezone.utc).isoformat(),'gold_availability':available}
        atomic_json(target/'run.json',state)
        try:
            for number,q in enumerate(questions,1):
                path=target/(q.question_id+'.json');row=read_json(path)
                if row:
                    if row['question_sha256']!=fingerprint(q.model_dump()) or row.get('profile_sha256')!=digest:
                        raise ValueError('Cached formula result differs from the current question/profile')
                    action='cached'
                else:
                    if pipeline is None:
                        if pipeline_factory is None:
                            from rag.pipeline import Pipeline
                            if retrieval_profile == 'v2':
                                from rag.pipeline_v2 import PipelineV2
                                build=lambda **kw:PipelineV2(settings,root=root,policy=policy,**kw)
                            else:
                                build=lambda **kw:Pipeline(settings,root=root,**kw)
                            if mode=='bm25':
                                class UnusedVectorStore:
                                    def close(self):pass
                                pipeline=build(store=UnusedVectorStore())
                            else:
                                pipeline=build()
                        else:
                            pipeline=pipeline_factory(settings,root)
                        live,_=pipeline.snapshot()
                        if live['snapshot_id']!=manifest['snapshot_id']:
                            raise ValueError('Index changed after preflight')
                        if mode in {'hybrid','dense'}:
                            pipeline.store.validate_collection()
                            if pipeline.store.count(manifest['snapshot_id'])!=manifest['chunks']:
                                raise ValueError('Qdrant count differs from the complete text snapshot')
                    if sha((root/'runtime/formulas/CURRENT.json').read_bytes())!=profile['formula_registry_sha256']:
                        raise ValueError('Authoritative formula registry changed during evaluation')
                    row=evaluate_question(q,gold,pipeline,store,manifest,available,mode=mode,rerank=rerank,
                                           top_k=top_k,formula_limit=formula_limit,english_lexical=english_lexical)
                    row['profile_sha256']=digest;atomic_json(path,row);action='new'
                rows.append(row)
                if emit:emit(number,len(questions),q,row,action)
            summary=summarize(rows);summary.update(profile_sha256=digest,questions=len(rows),annotation_human_verified=False,
                                                    limitations=metadata['limitations'])
            atomic_json(target/'summary.json',summary)
            atomic_json(target/'run.json',{**state,'status':'completed','finished_at':datetime.now(timezone.utc).isoformat()})
            return target,summary
        except BaseException:
            atomic_json(target/'run.json',{**state,'status':'interrupted_or_failed_partial_results_retained'})
            raise
        finally:
            if pipeline:
                pipeline.store.close()
