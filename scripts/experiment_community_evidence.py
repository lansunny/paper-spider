"""A: full text; B: dimension/claim anchors; C: community/claim anchors. No generation."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import importlib.metadata
from pathlib import Path
import random
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rag.benchmark import load_source_pages,score_evidence
from rag.community_evaluation import VARIANTS,run_case,summarize
from rag.community_retrieval import (VERSION,CommunityOriginalRetriever,CommunityPolicy,OriginalSources,load_claim_vectors)
from rag.embeddings import LocalModels
from rag.formula_store import digest_file
from rag.graph_evaluation import covered_units,scoped_pipeline,select_cases
from rag.ingestion import current_chunks,current_manifest,fingerprint
from rag.pipeline_v2 import PipelineV2,load_v2_settings
from rag.query_planning_v2 import PaperRouterV2
from rag.retrieval import rerank_candidates
from rag.reviews.corpus import ReviewCorpus
from rag.storage import atomic_json,read_json
from rag.vector_store import QdrantStore


class QueryCachedModels(LocalModels):
    def __init__(self,settings,root):
        super().__init__(settings,root);self.query_cache={}
    def query_vector(self,question):
        if question not in self.query_cache:self.query_cache[question]=super().query_vector(question)
        return self.query_cache[question].copy()


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--datasets',nargs='+',choices=['dev','test50'],default=['dev','test50'])
    parser.add_argument('--corpus',choices=['full','graph50'],default='full')
    parser.add_argument('--dimension',choices=['auto','topic','method','experiment','conclusion'],default='auto')
    parser.add_argument('--question-ids',nargs='+')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.limit<0:parser.error('limit must be nonnegative')
    cfg=load_v2_settings(args.config).model_copy(update={'rerank_k':40})
    policy=CommunityPolicy(dimension=args.dimension,context_chars=cfg.context_char_budget)
    corpus=ReviewCorpus(ROOT);manifest=current_manifest(cfg,ROOT)
    if not manifest or manifest['profile']!=cfg.index_profile():raise ValueError('Active index/profile mismatch')
    if any(manifest['documents'].get(p)!=v['document_sha256'] for p,v in corpus.papers.items()):
        raise ValueError('Graph and body indexes must use the same PDF versions')
    dataset_paths={s:ROOT/'data/evaluation/paper_benchmark'/(s+'.jsonl') for s in args.datasets}
    datasets={s:[__import__('json').loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l] for s,p in dataset_paths.items()}
    cases,excluded=select_cases(datasets,corpus.papers)
    if args.question_ids:
        if set(args.question_ids)-{c['question']['question_id'] for c in cases}:raise ValueError('Requested IDs are not eligible in the graph pilot')
        cases=[c for c in cases if c['question']['question_id'] in args.question_ids]
    if args.limit:cases=cases[:args.limit]
    if not cases:raise ValueError('No eligible source-grounded questions')
    allowed=sorted(corpus.papers if args.corpus=='graph50' else manifest['documents'])
    router=PaperRouterV2.from_project(allowed,ROOT)
    code=['rag/community_retrieval.py','rag/community_evaluation.py','scripts/experiment_community_evidence.py',
          'rag/pipeline_v2.py','rag/query_planning_v2.py','rag/retrieval.py','rag/lexical_v2.py',
          'rag/reviews/corpus.py','rag/paper_graph/retrieval.py','rag/benchmark.py','rag/graph_evaluation.py',
          'rag/embeddings.py','rag/chunking.py','rag/schema.py','configs/lexical_terms_zh_en.json']
    design={'version':VERSION,'graph_build_id':corpus.build_id,'graph_manifest_sha256':digest_file(corpus.directory/'manifest.json'),
        'index_snapshot_id':manifest['snapshot_id'],'retrieval_settings':cfg.model_dump(),'policy':asdict(policy),
        'retrieval_corpus':args.corpus,'retrieval_paper_ids':allowed,'graph_paper_ids':sorted(corpus.papers),
        'eligible_questions':[{'dataset':c['dataset'],'id':c['question']['question_id']} for c in cases],
        'dataset_sha256':{s:digest_file(p) for s,p in dataset_paths.items()},'code_sha256':{p:digest_file(ROOT/p) for p in code},
        'seed':args.seed,'variants':{'A':'fulltext','B':'fulltext + direct dimension paper selection + claim anchors',
                                  'C':'fulltext + community centroid gating + paper selection + claim anchors'},
        'controls':{'fulltext':'unchanged V2 query-only routing, lexical/dense pools and explicit-paper facets',
            'common_fulltext_order':'V2 balanced candidate prefix then the remaining complete fused pool',
            'before_cutoff_merge':'canonical source dedup + equal-weight RRF; no reserved tail slots',
            'candidate_preselection':40,'final_cross_encoder_pairs_per_arm':40,
            'cross_encoder_query':'original user question, no facet rescoring in any arm',
            'cross_encoder_text':'title + exact original text; anchors split with the body token-window policy',
            'claim_ranking':'shared local BGE-M3 vectors; no claim cross-encoder or online LLM',
            'B_C_difference':'only community gating; identical dimensions, descriptions, claims, scoring and source windows',
            'community_scores':'normalized mean of eligible member dimension vectors',
            'single_paper':'all arms reuse A exactly, source scope does not widen',
            'multi_paper_anchor_scope':'unambiguous question-only explicit paper facets',
            'context':'same exact JSON-list character budget, greedy whole novel fragments after source overlap subtraction',
            'dense_search':'exact','generation_api_calls':0,'warmup_excluded':True,
            'order':'seeded balanced arm rotations; fulltext retrieval shared once per question',
            'latency':'shared prefix + measured serial branch/rerank/packing stages; not an independent end-to-end load test'},
        'primary_endpoint':'context evidence-unit coverage on non-single-paper routes',
        'secondary_endpoints':['new/lost required evidence units','complete evidence rate','paper recall','source-bank ceiling','latency and actual CE pairs'],
        'limitations':['Previously used dev/test questions: diagnostic regression, not untouched heldout data.',
            'All positive anchors for eligible questions are within the existing 50 graph papers; no claim of full-library generalization.',
            'Agent-authored labels are source-checked but not independently human-verified; unlabelled passages are not assumed irrelevant.',
            'Community centroids test membership organization, not citation relations or causal knowledge edges.',
            'Offline claim encoding is extra cost, separately reported; equal final rerank ceilings do not imply equal total retrieval cost.',
            'No answer generation, factual answer accuracy or refusal quality is measured.',
            'Connected-gold-paper-group bootstrap; fewer than 10 groups is descriptive only.']}
    identity=fingerprint(design)
    target=(args.resume or ROOT/'runtime/experiments'/('community-evidence-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])).resolve()
    if not target.is_relative_to((ROOT/'runtime/experiments').resolve()):raise ValueError('Experiment output must stay under runtime/experiments')
    prior=read_json(target/'run.json')
    if prior and prior['design_sha256']!=identity:raise ValueError('Resume code/data/settings differ from this experiment')
    state={'status':'prepared','design_sha256':identity,'design':design,'excluded':excluded,
           'questions':len(cases),'routes':dict(Counter(router.route(c['question']['question'])['mode'] for c in cases))}
    atomic_json(target/'run.json',state)
    print({'retrieval_papers':len(allowed),'graph_papers':len(corpus.papers),'questions':len(cases),
           'routes':state['routes'],'rerank_limit':40,'context_chars':policy.context_chars,'generation_api_calls':0},flush=True)
    print('Output:',target,flush=True)
    if not args.run:
        print('Preview only. Add --run to execute A/B/C locally.',flush=True);return 0
    full,chunks=current_chunks(cfg,ROOT)
    if full['snapshot_id']!=manifest['snapshot_id']:raise ValueError('Index changed during preparation')
    models=QueryCachedModels(cfg,ROOT);store=QdrantStore(cfg)
    rows=[]
    try:
        store.validate_collection();warm=perf_counter()
        if args.corpus=='graph50':
            pipeline,lexical=scoped_pipeline(cfg,ROOT,models,store,manifest,chunks,allowed)
            local,_=pipeline.snapshot()
        else:
            pipeline=PipelineV2(cfg,root=ROOT,models=models,store=store);local,lexical=pipeline.snapshot()
        resolver=OriginalSources(ROOT,local,corpus)
        ids,vectors,cache=load_claim_vectors(corpus,models,ROOT,lambda **s:print(s,flush=True))
        retriever=CommunityOriginalRetriever(corpus,models,ids,vectors,policy)
        models.query_vector('Research retrieval evaluation')
        rerank_candidates(models,'Research retrieval evaluation',[lexical.chunks[0].payload()],title=True)
        evaluation_papers={s['paper_id'] for c in cases for u in c['question']['evidence_units'] for a in u['alternatives'] for s in a}
        pages=load_source_pages(evaluation_papers,ROOT)
        all_anchors=[]
        for sid in corpus.sources:
            s=corpus.source(sid)
            all_anchors.append({'chunk_id':sid,'paper_id':s['paper_id'],'document_sha256':s['document_sha256'],
                'pdf_page':s['pdf_page'],'title':s['title'],'text':s['text'],'source_spans':[{'block_number':-1,
                    'source_start':s['source_start'],'source_end':s['source_end'],'text_start':0,'text_end':len(s['text'])}]})
        ceilings={c['question']['question_id']:score_evidence(c['question'],all_anchors,pages,k=len(all_anchors)) for c in cases}
        state.update(status='running',claim_vector_cache=cache,warmup_ms=round((perf_counter()-warm)*1000,2),
                     device=models._device(),dependencies={p:importlib.metadata.version(p) for p in ('numpy','torch','sentence-transformers','qdrant-client')})
        atomic_json(target/'run.json',state);atomic_json(target/'anchor-bank-ceiling.json',ceilings)
        execution=list(cases);rng=random.Random(args.seed);rng.shuffle(execution);rotation=list(VARIANTS);rng.shuffle(rotation)
        for index,case in enumerate(execution):
            q=case['question'];path=target/'cases'/(q['question_id']+'.json');cached=read_json(path)
            if cached:
                if cached['design_sha256']!=identity:raise ValueError('Cached case belongs to another experiment')
                rows.append(cached);continue
            if current_manifest(cfg,ROOT)['snapshot_id']!=manifest['snapshot_id']:raise ValueError('Active index changed')
            corpus.assert_current()
            order=[rotation[(index+j)%3] for j in range(3)]
            row=run_case(case,pipeline,retriever,resolver,models,policy,pages,order)
            row.update(design_sha256=identity,anchor_bank_ceiling=ceilings[q['question_id']])
            row['anchor_bank_units_absent_from_A_pool']=sorted(covered_units(ceilings[q['question_id']])-covered_units(row['results']['A']['metrics']['candidate_pool']))
            atomic_json(path,row);rows.append(row)
            print(f"[{index+1}/{len(execution)}] {q['question_id']} "+' '.join(v+'='+format(row['results'][v]['metrics']['context_budget']['evidence_unit_recall'],'.1%') for v in VARIANTS)+(' protected' if row['single_paper_guard'] else ''),flush=True)
            state['completed_questions']=len(rows);atomic_json(target/'run.json',state)
        corpus.assert_current()
        for pid in corpus.papers:corpus.canonical_pages(pid)
        summary={'design_sha256':identity,'questions':len(rows),'retrieval_papers':len(allowed),'graph_papers':len(corpus.papers),
            'generation_api_calls':0,'groups':summarize(rows,cases,args.seed),
            'changed_cases':[{'question_id':r['question_id'],'question':r['question'],'changes':r['changes']} for r in rows
                             if any(x['gained_units'] or x['lost_units'] for d in r['changes'].values() for x in d.values())]}
        atomic_json(target/'summary.json',summary);state.update(status='completed',completed_questions=len(rows));atomic_json(target/'run.json',state)
        for scope in ['primary_non_single','single_paper_guard','all']:
            if scope not in summary['groups']:continue
            print(scope,flush=True)
            for v,values in summary['groups'][scope]['variants'].items():
                m=values['stages']['context_budget'];print(f"  {v}: coverage={m['evidence_coverage']:.1%}, complete={m['complete_evidence_rate']:.1%}, papers={m['paper_recall']:.1%}",flush=True)
        print('Summary:',target/'summary.json',flush=True)
    except BaseException as error:
        state.update(status='failed',completed_questions=len(rows),error_type=type(error).__name__,error=str(error)[:800]);atomic_json(target/'run.json',state);raise
    finally:store.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
