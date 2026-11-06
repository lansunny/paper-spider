"""Persisted related-reading jobs; no generation until the user starts a request."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.reviews import now
from rag.related_reading import RelatedReadingEngine, ReadingRequest
from rag.reviews.corpus import ReviewCorpus
from rag.storage import atomic_json, read_json


class RelatedReadingJobs:
    def __init__(self,root,pipeline_provider):
        self.root=Path(root);self.pipeline_provider=pipeline_provider
        self.directory=self.root/'runtime/related_reading/jobs'
        self.lock=Lock();self.active=None;self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='related-reading')
        self._corpus=None
        for path in self.directory.glob('*/job.json'):
            job=read_json(path)
            if job['status'] in {'queued','running'}:
                atomic_json(path,{**job,'status':'interrupted','updated_at':now(),'message':'任务已中断，可重新提交；已完成步骤会复用缓存。'})

    def corpus(self):
        if self._corpus:
            try:self._corpus.assert_current();return self._corpus
            except ValueError:self._corpus=None
        self._corpus=ReviewCorpus(self.root)
        return self._corpus

    def path(self,run_id):
        try:
            if str(UUID(run_id))!=run_id:raise ValueError()
        except ValueError:raise KeyError(run_id) from None
        return self.directory/run_id

    def start(self,request,enqueue=None):
        corpus=self.corpus()
        if request.paper_id not in corpus.papers:raise ValueError('这篇论文尚未纳入关联阅读图。')
        with self.lock:
            if self.active:raise RuntimeError('已有关联阅读任务正在处理，请稍后再试。')
            run_id=str(uuid4());directory=self.path(run_id)
            job={'run_id':run_id,'status':'queued','created_at':now(),'updated_at':now(),
                 'paper_id':request.paper_id,'goal':request.goal,'graph_build_id':corpus.build_id,
                 'progress':{'stage':'queued','completed':0,'total':1}}
            atomic_json(directory/'request.json',request.model_dump());atomic_json(directory/'job.json',job)
            self.active=run_id
            if enqueue:enqueue(self.executor.submit,self._run,run_id,request,corpus)
            else:self.executor.submit(self._run,run_id,request,corpus)
        return job

    def _run(self,run_id,request,corpus):
        directory=self.path(run_id);job=read_json(directory/'job.json')
        def progress(**state):
            job.update(status='running',updated_at=now(),progress=state)
            atomic_json(directory/'job.json',job)
        try:
            progress(stage='neighbors',completed=0,total=1)
            result=RelatedReadingEngine(self.root,corpus,self.pipeline_provider()).run(request,progress)
            result['run_id']=run_id
            atomic_json(directory/'result.json',result)
            job.update(status='completed',updated_at=now(),progress={'stage':'completed','completed':1,'total':1})
        except Exception as error:
            logging.getLogger(__name__).exception('Related reading failed: %s',run_id)
            job.update(status='failed',updated_at=now(),error_type=type(error).__name__,
                message=str(error)[:300] if isinstance(error,(ValueError,FileNotFoundError)) and any('\u4e00'<=c<='\u9fff' for c in str(error))
                else '关联阅读未完成，可重新提交；已完成的模型步骤会复用缓存。')
        finally:
            atomic_json(directory/'job.json',job)
            with self.lock:self.active=None

    def read(self,run_id):
        directory=self.path(run_id);job=read_json(directory/'job.json')
        if not job:raise KeyError(run_id)
        if job['status']=='completed':
            result=read_json(directory/'result.json')
            corpus=self.corpus()
            if result['graph_build_id']!=corpus.build_id:raise ValueError('论文图已更新，请重新生成关联阅读。')
            for pid in {e['paper_id'] for e in result['evidence']}:corpus.canonical_pages(pid)
            return {**job,'result':result}
        return job


def create_related_reading_router(root,pipeline_provider):
    router=APIRouter(prefix='/api/related-reading',tags=['related-reading'])
    jobs=RelatedReadingJobs(root,pipeline_provider)
    def read(action):
        try:return action()
        except (KeyError,FileNotFoundError):raise HTTPException(404,'关联阅读记录或论文尚不存在。') from None
        except (ValueError,RuntimeError) as error:raise HTTPException(409,str(error)) from None

    @router.get('/papers/{paper_id}')
    def paper(paper_id:str):
        corpus=read(jobs.corpus)
        return {'available':paper_id in corpus.papers,'papers':len(corpus.papers),'graph_build_id':corpus.build_id,
                'paper':corpus.papers.get(paper_id),'active_run_id':jobs.active}

    @router.get('/jobs')
    def recent(paper_id:str|None=None):
        values=[read_json(p) for p in jobs.directory.glob('*/job.json')]
        return {'jobs':sorted([j for j in values if paper_id is None or j['paper_id']==paper_id],
            key=lambda j:j['created_at'],reverse=True)[:12]}

    @router.post('')
    def start(request:ReadingRequest,background_tasks:BackgroundTasks):
        request.goal=request.goal.strip()
        if len(request.goal)<2:raise HTTPException(422,'请填写希望对照学习的问题。')
        return read(lambda:jobs.start(request,background_tasks.add_task))

    @router.get('/jobs/{run_id}')
    def job(run_id:str):return read(lambda:jobs.read(run_id))
    return router
