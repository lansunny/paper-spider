"""Opt-in Docling retrieval routes; the existing answer endpoint keeps its index."""
from functools import lru_cache
import os
from typing import Annotated, Literal
from urllib.parse import quote, urlencode

from fastapi import APIRouter, File, Form, HTTPException, Path, Query, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag.docling_search import DoclingSearch, QueryLanguage, profiles
from rag.figure_vectors import MAX_IMAGE_BYTES

router = APIRouter(prefix='/api/docling', tags=['Docling retrieval'])
Profile = Annotated[str, Query(pattern=r'^[a-f0-9]{16}$')]
ObjectId = Annotated[str, Path(pattern=r'^[a-f0-9]{32}$')]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=1500)
    profile: str | None = Field(default=None, pattern=r'^[a-f0-9]{16}$')
    pilot: bool = False
    paper_id: str | None = Field(default=None, min_length=1, max_length=160)
    paper_ids: list[Annotated[str, Field(min_length=1, max_length=160)]] = Field(default_factory=list, max_length=100)
    kinds: set[Literal['body', 'figure', 'table']] = Field(default_factory=lambda: {'body','figure','table'}, min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    clip_query: str | None = Field(default=None, min_length=1, max_length=350)
    visual: bool = True
    nearby: bool = True
    query_language: QueryLanguage | None = 'english'

    @model_validator(mode='after')
    def pilot_is_explicit(self):
        if self.pilot and not self.profile:
            raise ValueError('Choose a profile for pilot queries')
        if self.paper_id and self.paper_ids:
            raise ValueError('Choose paper_id or paper_ids')
        return self


@lru_cache(maxsize=2)
def _engine(profile, pilot, snapshot):
    engine = DoclingSearch(profile, pilot=pilot)
    if engine.manifest['snapshot_id'] != snapshot:
        raise RuntimeError('Docling index switched; retry the request')
    return engine


def get_engine(profile=None, pilot=False):
    if profile is None and not pilot:
        profile = os.environ.get('PAPERRAG_DOCLING_PROFILE') or None
    candidates = [p for p in profiles() if p['pilot'] == pilot and (profile is None or p['profile'] == profile)]
    if not candidates:
        raise HTTPException(503, '多模态索引尚未就绪。')
    if len(candidates) > 1:
        raise HTTPException(422, '存在多个多模态索引，请指定要使用的配置。')
    p = candidates[0]
    return _engine(p['profile'], pilot, p['snapshot_id'])


def _links(obj, profile, pilot, snapshot):
    paper = quote(obj['paper_id'], safe='')
    oid = obj['object_id']
    args = urlencode({'profile': profile, 'pilot': str(pilot).lower(), 'snapshot': snapshot})
    page = obj.get('matched_page') or (obj['pages'][0] if obj['pages'] else None)
    obj['pdf_url'] = f'/api/papers/{paper}/pdf' + ('#page='+str(page) if page else '')
    obj['object_url'] = f'/api/docling/objects/{oid}?{args}'
    for i, crop in enumerate(obj['crops']):
        crop['image_url'] = f'/api/docling/objects/{oid}/crops/{i}?{args}'
    for paragraph in obj.get('related_paragraphs', []):
        _links(paragraph, profile, pilot, snapshot)
    if paragraph := obj.get('nearby_paragraph'):
        _links(paragraph, profile, pilot, snapshot)


def _respond(engine, result):
    for group in ('body', 'body_context', 'figures', 'tables'):
        for obj in result.get(group, ()):
            _links(obj, engine.profile, engine.pilot, result['snapshot_id'])
            if group in {'body', 'body_context'}:
                suffix = urlencode({'profile': engine.profile, 'pilot': str(engine.pilot).lower(), 'snapshot': result['snapshot_id']})
                obj['linked_objects'] = [{'object_id': oid, 'object_url': f'/api/docling/objects/{oid}?{suffix}'}
                                         for oid in obj['linked_object_ids']]
    return result


@router.get('/profiles')
def list_profiles():
    return {'profiles': profiles(), 'default_profile': os.environ.get('PAPERRAG_DOCLING_PROFILE') or None}


@router.post('/search')
def search_documents(request: SearchRequest):
    try:
        engine = get_engine(request.profile, request.pilot)
        result = engine.search(request.query, paper_id=request.paper_id, paper_ids=request.paper_ids, kinds=request.kinds,
                               top_k=request.top_k, clip_text=request.clip_query, visual=request.visual, nearby=request.nearby,
                               query_language=request.query_language or 'english')
        return _respond(engine, result)
    except KeyError as error:
        raise HTTPException(404, '选中的论文尚未进入多模态索引。') from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    except (ResponseHandlingException, UnexpectedResponse) as error:
        raise HTTPException(503, 'Docling vector search is temporarily unavailable') from error


@router.post('/search/image')
def search_image(image: Annotated[UploadFile, File()],
                 profile: Annotated[str, Form(pattern=r'^[a-f0-9]{16}$')],
                 pilot: Annotated[bool, Form()] = False,
                 paper_id: Annotated[str | None, Form(min_length=1, max_length=160)] = None,
                 paper_ids: Annotated[list[Annotated[str, Field(min_length=1, max_length=160)]] | None, Form(max_length=100)] = None,
                 top_k: Annotated[int, Form(ge=1, le=20)] = 5):
    try:
        engine = get_engine(profile, pilot)
        blob = image.file.read(MAX_IMAGE_BYTES+1)
        if paper_id and paper_ids:
            raise ValueError('请选择一种论文范围。')
        return _respond(engine, engine.search(paper_id=paper_id, paper_ids=paper_ids, kinds=('figure',),
                                             top_k=top_k, query_image=blob))
    except KeyError as error:
        raise HTTPException(404, '选中的论文尚未进入多模态索引。') from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    finally:
        image.file.close()


def _source_engine(profile, pilot, snapshot):
    engine = get_engine(profile, pilot)
    if snapshot and engine.manifest['snapshot_id'] != snapshot:
        raise HTTPException(409, 'This search result belongs to another index version; search again')
    return engine


@router.get('/objects/{object_id}')
def get_object(object_id: ObjectId, profile: Profile, pilot: bool = False,
               snapshot: Annotated[str | None, Query(pattern=r'^[a-f0-9]{32}$')] = None):
    try:
        engine = _source_engine(profile, pilot, snapshot)
        obj = engine.load_object(object_id)
        _links(obj, profile, pilot, engine.manifest['snapshot_id'])
        return obj
    except KeyError as error:
        raise HTTPException(404, 'Object not found in this snapshot') from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error


@router.get('/objects/{object_id}/crops/{index}')
def get_crop(object_id: ObjectId, index: Annotated[int, Path(ge=0)], profile: Profile, pilot: bool = False,
             snapshot: Annotated[str | None, Query(pattern=r'^[a-f0-9]{32}$')] = None):
    try:
        engine = _source_engine(profile, pilot, snapshot)
        return Response(engine.crop(object_id, index), media_type='image/png')
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(404, 'Crop not found in this snapshot') from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
