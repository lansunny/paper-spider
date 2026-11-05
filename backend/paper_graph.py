"""Read-only access to the current four-view paper graph and exact source anchors."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from rag.paper_graph.store import PaperGraphStore

Dimension = Literal['topic', 'method', 'experiment', 'conclusion']


def create_paper_graph_router(root):
    router = APIRouter(prefix='/api/paper-graph', tags=['paper-graph'])
    store = PaperGraphStore(root)

    def read(action):
        try:
            return action()
        except (FileNotFoundError, KeyError):
            raise HTTPException(404, '论文图或对应论文、来源尚不存在。') from None
        except ValueError:
            raise HTTPException(409, '论文图版本校验失败，请重新构建。') from None

    @router.get('')
    def summary():
        value = read(store.summary)
        return {**value, 'retrieval_integration_at_build': value.get('retrieval_integration'),
                'retrieval_integration': 'opt_in_one_hop', 'retrieval_enabled_by_default': False}

    @router.get('/views/{dimension}')
    def view(dimension: Dimension):
        return read(lambda: store.view(dimension))

    @router.get('/sources/{source_id}')
    def source(source_id: str):
        return read(lambda: store.source(source_id))

    @router.get('/papers/{paper_id}')
    def paper(paper_id: str):
        return read(lambda: store.paper(paper_id))

    @router.get('/papers/{paper_id}/neighbors')
    def neighbors(paper_id: str, dimension: Dimension = 'topic', limit: int = Query(10, ge=1, le=100)):
        return read(lambda: {'paper_id': paper_id, 'dimension': dimension,
                             'neighbors': store.neighbors(paper_id, dimension, limit)})

    return router
