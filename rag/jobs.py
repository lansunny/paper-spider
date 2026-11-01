"""A single local indexing worker with durable, pollable progress."""
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from uuid import uuid4

from rag.ingestion import build_index
from rag.storage import atomic_json, read_json


class IndexJobs:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.path = pipeline.root / 'runtime/index_job.json'
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='paperrag-index')
        self.lock = Lock()
        self.future = None

    def status(self):
        state = read_json(self.path, {'status': 'idle'})
        if state['status'] in {'queued', 'running'} and state.get('pid') != os.getpid():
            return {**state, 'status': 'interrupted', 'message': '上次服务已退出，可重新建立索引。'}
        return state

    def start(self):
        with self.lock:
            if self.future and not self.future.done():
                return self.status()
            state = {'job_id': uuid4().hex, 'pid': os.getpid(), 'status': 'queued',
                     'stage': 'starting', 'completed': 0, 'total': 0}
            atomic_json(self.path, state)

            def work():
                def progress(**changes):
                    state.update(status='running', **changes)
                    atomic_json(self.path, state)
                try:
                    p = self.pipeline
                    manifest = build_index(p.settings, p.models, p.store, p.root, progress=progress)
                    state.update(status='completed', stage='ready', completed=manifest['chunks'],
                                 total=manifest['chunks'], snapshot_id=manifest['snapshot_id'])
                except Exception as error:
                    state.update(status='failed', message=f'{type(error).__name__}: {str(error)[:400]}')
                atomic_json(self.path, state)

            self.future = self.executor.submit(work)
            return dict(state)
