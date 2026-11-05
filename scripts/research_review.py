"""Run a sourced research review from the current graph and prepared cards."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.rag_service import get_pipeline
from rag.reviews.corpus import ReviewCorpus
from rag.reviews.engine import ReviewEngine
from rag.reviews.schema import ReviewRequest
from rag.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--scope', choices=['topic','library'], default='topic')
    parser.add_argument('--evidence-only', action='store_true')
    args = parser.parse_args()
    from uuid import uuid4
    run_id = str(uuid4())
    directory = ROOT/'runtime/reviews/jobs'/run_id
    request = ReviewRequest(question=args.question,scope=args.scope,generate=not args.evidence_only)
    from backend.reviews import now
    job = {'run_id':run_id,'kind':'review','status':'running','created_at':now(),'updated_at':now(),
           'question':request.question,'progress':{'stage':'starting','completed':0,'total':1}}
    atomic_json(directory/'request.json', request.model_dump())
    atomic_json(directory/'job.json', job)
    def progress(**state):
        print(state, flush=True)
        atomic_json(directory/'job.json',{**job,'updated_at':now(),'progress':state})
    try:
        result = ReviewEngine(ROOT,ReviewCorpus(ROOT),get_pipeline()).run(request,progress)
        result['run_id'] = run_id
        result['map_url'] = '/graph?dimension=topic&review='+run_id
        for route in result['routes']:
            route['map_url'] += '&review='+run_id+'&route='+route['route_id']
        atomic_json(directory/'result.json',result)
        atomic_json(directory/'job.json',{**job,'status':'completed','updated_at':now(),
                    'progress':{'stage':'completed','completed':1,'total':1}})
        print({'run_id':run_id,'routes':len(result['routes']),'evidence':len(result['evidence']),
               'tokens':result['usage'],'result':str(directory/'result.json')},flush=True)
    except Exception as error:
        atomic_json(directory/'job.json',{**job,'status':'failed','error_type':type(error).__name__,
                    'updated_at':now(),'message':'综述未完成，可检查日志后重试。'})
        raise


if __name__ == '__main__':
    main()
