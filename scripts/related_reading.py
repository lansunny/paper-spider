"""Preview graph neighbors or run the sourced related-reading assistant."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from backend.rag_service import get_pipeline
from backend.related_reading import RelatedReadingJobs
from rag.related_reading import DEFAULT_GOAL, ReadingRequest, graph_candidates
from rag.reviews.corpus import ReviewCorpus


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paper_id')
    parser.add_argument('--goal',default=DEFAULT_GOAL)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--evidence-only',action='store_true')
    args=parser.parse_args()
    if not args.run:
        corpus=ReviewCorpus(ROOT)
        print(json.dumps({'graph_build_id':corpus.build_id,'papers':len(corpus.papers),
            'candidates':graph_candidates(corpus,args.paper_id,args.goal),
            'generation_api_calls':0,'preview_only':True},ensure_ascii=False,indent=2))
        return 0
    jobs=RelatedReadingJobs(ROOT,get_pipeline)
    job=jobs.start(ReadingRequest(paper_id=args.paper_id,goal=args.goal,generate=not args.evidence_only))
    print('Run:',job['run_id'],flush=True)
    previous=None
    try:
        while job['status'] in {'queued','running'}:
            state=job.get('progress',{})
            if state!=previous:
                print(json.dumps(state,ensure_ascii=False),flush=True);previous=state
            time.sleep(.5);job=jobs.read(job['run_id'])
        if job['status']!='completed':
            print(job.get('message',job['status']),flush=True)
            return 1
        result=job['result']
        print(json.dumps({'papers':result['corpus_papers'],'recommendations':len(result['recommendations']),
            'supported':sum(r['status']=='supported' for r in result['recommendations']),
            'evidence':len(result['evidence']),'model_calls':result['online_model_calls'],'usage':result['usage'],
            'result':str(jobs.path(job['run_id'])/'result.json')},ensure_ascii=False),flush=True)
        return 0
    finally:jobs.executor.shutdown(wait=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
