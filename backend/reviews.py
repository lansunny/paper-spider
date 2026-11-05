"""Persistent, single-worker research reviews over the current paper graph."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException

from rag.reviews.cards import build_cards, load_cards
from rag.reviews.corpus import ReviewCorpus
from rag.reviews.engine import ReviewEngine
from rag.reviews.schema import ReviewRequest
from rag.storage import atomic_json, read_json


def now():
    return datetime.now(timezone.utc).isoformat()


class ReviewJobs:
    def __init__(self, root, pipeline_provider):
        self.root = Path(root)
        self.directory = self.root/'runtime/reviews/jobs'
        self.pipeline_provider = pipeline_provider
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='research-review')
        self.active = None
        self._corpus = None
        for path in self.directory.glob('*/job.json'):
            job = read_json(path)
            if job['status'] in {'queued', 'running'}:
                atomic_json(path, {**job, 'status':'interrupted', 'updated_at':now(),
                                  'message':'服务已重启。可重新提交，已完成的模型步骤会复用缓存。'})

    def corpus(self):
        corpus = self._corpus
        if corpus:
            try:
                corpus.assert_current()
                return corpus
            except ValueError:
                self._corpus = None
        self._corpus = ReviewCorpus(self.root)
        return self._corpus

    def path(self, run_id):
        try:
            if str(UUID(run_id)) != run_id:
                raise ValueError('Invalid review ID')
        except ValueError:
            raise KeyError(run_id) from None
        return self.directory/run_id

    def read(self, run_id, validate=True):
        directory = self.path(run_id)
        job = read_json(directory/'job.json')
        if not job:
            raise KeyError(run_id)
        if job['status'] == 'completed':
            result = read_json(directory/'result.json')
            if validate and job['kind'] == 'review':
                corpus = self.corpus()
                if result['graph_build_id'] != corpus.build_id:
                    raise ValueError('论文图已更新，此综述基于旧图，请重新生成。')
                for paper_id in result['map_trace']['paper_ids']:
                    corpus.canonical_pages(paper_id)
            return {**job, 'result':result}
        return job

    def start(self, request=None, kind='review', enqueue=None):
        if kind == 'review':
            corpus = self.corpus()
            load_cards(self.root, corpus)
        with self.lock:
            if self.active:
                raise RuntimeError('已有综述任务正在处理，请等待当前任务完成。')
            run_id = str(uuid4())
            job = {'run_id':run_id, 'kind':kind, 'status':'queued', 'created_at':now(),
                   'updated_at':now(), 'progress':{'stage':'queued','completed':0,'total':1}}
            if request:
                atomic_json(self.path(run_id)/'request.json', request.model_dump())
                job['question'] = request.question
            atomic_json(self.path(run_id)/'job.json', job)
            self.active = run_id
            if enqueue:
                enqueue(self.executor.submit, self._run, run_id, request, kind)
            else:
                self.executor.submit(self._run, run_id, request, kind)
        return job

    def _run(self, run_id, request, kind):
        directory = self.path(run_id)
        job = read_json(directory/'job.json')
        def progress(**state):
            job.update(status='running', updated_at=now(), progress=state)
            atomic_json(directory/'job.json', job)
        try:
            progress(stage='starting', completed=0, total=1)
            if kind == 'cards':
                result = build_cards(self.root, progress)
            else:
                result = ReviewEngine(self.root, self.corpus(), self.pipeline_provider()).run(request, progress)
                result['run_id'] = run_id
                result['map_url'] = '/graph?dimension=topic&review='+run_id
                for route in result['routes']:
                    route['map_url'] += '&review='+run_id+'&route='+route['route_id']
            atomic_json(directory/'result.json', result)
            job.update(status='completed', updated_at=now(), progress={'stage':'completed','completed':1,'total':1})
        except Exception as error:
            logging.getLogger(__name__).exception('Research review failed: %s', run_id)
            job.update(status='failed', updated_at=now(), error_type=type(error).__name__,
                       message=str(error)[:300] if isinstance(error, (ValueError, FileNotFoundError)) else
                       '综述未完成，请检查本地模型、生成服务与论文来源。已完成步骤会保留缓存。')
        finally:
            atomic_json(directory/'job.json', job)
            with self.lock:
                self.active = None


def create_reviews_router(root, pipeline_provider):
    router = APIRouter(prefix='/api/reviews', tags=['research-reviews'])
    jobs = ReviewJobs(root, pipeline_provider)

    def read(action):
        try:
            return action()
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, '研究卡片、原文或综述记录尚不存在。') from None
        except ValueError as error:
            raise HTTPException(409, str(error)) from None
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from None

    @router.get('/status')
    def status():
        corpus = read(jobs.corpus)
        try:
            manifest, cards = load_cards(root, corpus)
        except (FileNotFoundError, ValueError) as error:
            return {'ready':False,'papers':len(corpus.papers),'message':str(error),'active_run_id':jobs.active}
        return {'ready':True,'papers':len(corpus.papers),'topic_cards':len(cards),
                'method_groups':len(corpus.communities['method']),'card_routes':manifest['method_routes'],
                'graph_build_id':corpus.build_id,'cardset_id':manifest['cardset_id'],'active_run_id':jobs.active}

    @router.get('/cards')
    def cards():
        corpus = read(jobs.corpus)
        manifest, values = read(lambda:load_cards(root, corpus))
        return {'cards':values,'manifest':manifest,
                'papers':[{k:v for k,v in p.items() if k in {'paper_id','title','year'}} for p in corpus.papers.values()]}

    @router.get('/sources/{source_id}')
    def source(source_id: str):
        return read(lambda:jobs.corpus().source(source_id))

    @router.get('/jobs')
    def recent():
        values = [read_json(p) for p in jobs.directory.glob('*/job.json')]
        values = [j for j in values if j.get('visibility') != 'diagnostic']
        return {'jobs':sorted(values, key=lambda j:j['created_at'], reverse=True)[:12]}

    @router.post('/cards')
    def prepare_cards(background_tasks: BackgroundTasks):
        return read(lambda:jobs.start(kind='cards',enqueue=background_tasks.add_task))

    @router.post('')
    def start(request: ReviewRequest, background_tasks: BackgroundTasks):
        request.question = request.question.strip()
        if len(request.question) < 2:
            raise HTTPException(422, '请输入研究问题。')
        return read(lambda:jobs.start(request,enqueue=background_tasks.add_task))

    @router.get('/jobs/{run_id}')
    def job(run_id: str):
        return read(lambda:jobs.read(run_id))

    return router
