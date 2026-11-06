"""Local-only FastAPI shell: library, PDF upload/view and a replaceable RAG endpoint."""
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import quote

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import rag_service
from rag.formula_store import FormulaStore

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / 'runtime/uploads'
UPLOAD_MANIFEST = UPLOADS / 'manifest.json'
UPLOAD_LOCK = Lock()
MAX_FILE_BYTES = 30 * 1024 * 1024

app = FastAPI(title='Paper Spider', version='0.1.0')


def read_uploaded():
    return json.loads(UPLOAD_MANIFEST.read_text(encoding='utf-8')) if UPLOAD_MANIFEST.exists() else []


def papers(index_status=True):
    originals = json.loads((ROOT / 'data/paper_manifest.json').read_text(encoding='utf-8'))
    result = []
    for paper in originals:
        result.append({
            'paper_id': paper['paper_id'], 'title': paper['title'],
            'year': paper.get('publication_year'), 'pdf_path': paper['pdf_path'],
            'source_url': paper['landing_url'], 'sha256': paper['sha256'],
            'indexed': (ROOT / 'data/parsed' / (paper['paper_id'] + '.jsonl')).exists(),
            'origin': 'curated',
        })
    with UPLOAD_LOCK:
        result.extend(read_uploaded())
    if index_status:
        indexed = rag_service.status()['indexed_documents']
        for paper in result:
            paper['indexed'] = indexed.get(paper['paper_id']) == paper['sha256']
    return result


def public_paper(paper):
    return {**{k: v for k, v in paper.items() if k not in {'pdf_path', 'sha256'}},
            'pdf_url': '/api/papers/' + quote(paper['paper_id'], safe='') + '/pdf'}


@app.get('/api/health')
def health():
    library = papers()
    state = rag_service.status()
    state.pop('indexed_documents')
    return {'status': 'ok', **state, 'papers': len(library),
            'indexed_papers': sum(p['indexed'] for p in library)}


@app.post('/api/index')
def start_index():
    return rag_service.get_jobs().start()


@app.get('/api/index')
def index_status():
    return rag_service.get_jobs().status()


@app.get('/api/papers')
def list_papers():
    return {'papers': [public_paper(p) for p in papers()]}


@app.get('/api/papers/{paper_id}/pdf')
def paper_pdf(paper_id: str):
    paper = next((p for p in papers(False) if p['paper_id'] == paper_id), None)
    if not paper:
        raise HTTPException(404, '论文不存在。')
    path = (ROOT / paper['pdf_path']).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise HTTPException(404, 'PDF 文件不存在。')
    return FileResponse(path, media_type='application/pdf', headers={'Content-Disposition': 'inline'})


@app.post('/api/papers/upload')
def upload_paper(file: UploadFile = File(...)):
    name = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
    if not name.lower().endswith('.pdf'):
        raise HTTPException(400, '只支持 PDF 文件。')
    content = bytearray()
    while chunk := file.file.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(413, '单个 PDF 不能超过 30 MB。')
    if not content.startswith(b'%PDF-'):
        raise HTTPException(400, '文件不是有效 PDF。')
    try:
        with pymupdf.open(stream=bytes(content), filetype='pdf') as pdf:
            if pdf.needs_pass or not len(pdf):
                raise ValueError('Encrypted or empty PDF')
            page_count = len(pdf)
    except Exception:
        raise HTTPException(400, '无法读取该 PDF，请确认文件完整且未加密。') from None
    digest = hashlib.sha256(content).hexdigest()
    existing = next((p for p in papers() if p['sha256'] == digest), None)
    if existing:
        return {'paper': public_paper(existing), 'duplicate': True, 'message': '相同文件已在论文库中。'}
    identifier = 'upload-' + digest
    with UPLOAD_LOCK:
        uploads = read_uploaded()
        existing = next((p for p in uploads if p['sha256'] == digest), None)
        if existing:
            return {'paper': public_paper(existing), 'duplicate': True, 'message': '相同文件已在论文库中。'}
        UPLOADS.mkdir(parents=True, exist_ok=True)
        target = UPLOADS / (digest + '.pdf')
        target.write_bytes(content)
        record = {'paper_id': identifier, 'title': name[:250], 'year': None,
                  'pdf_path': target.relative_to(ROOT).as_posix(), 'sha256': digest,
                  'source_url': None, 'indexed': False, 'origin': 'uploaded', 'pages': page_count}
        uploads.append(record)
        temporary = UPLOAD_MANIFEST.with_suffix('.tmp')
        temporary.write_text(json.dumps(uploads, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(UPLOAD_MANIFEST)
    return {'paper': public_paper(record), 'duplicate': False,
            'message': 'PDF 已保存。点击“建立/更新索引”完成解析、分块和向量入库。'}


@app.get('/api/papers/{paper_id}/pages/{page_number}')
def paper_page(paper_id: str, page_number: int):
    response = paper_pdf(paper_id)
    with pymupdf.open(response.path) as pdf:
        if page_number < 1 or page_number > len(pdf):
            raise HTTPException(404, '页码超出范围。')
        page = pdf[page_number - 1]
        scale = 1400 / max(page.rect.width, page.rect.height)
        png = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes('png')
        return Response(png, media_type='image/png', headers={'X-Page-Count': str(len(pdf))})


@app.get('/api/formulas/{paper_id}/{element_id}/image')
def formula_image(paper_id: str, element_id: str):
    try:
        path = FormulaStore(ROOT).image(paper_id, element_id)
    except KeyError:
        raise HTTPException(404, '公式不属于当前权威产物。') from None
    except (ValueError, OSError):
        raise HTTPException(409, '公式原图校验失败，请重新抽取。') from None
    return FileResponse(path, media_type='image/png')


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    paper_ids: list[str] = Field(default_factory=list, max_length=100)
    top_k: int = Field(default=5, ge=1, le=10)
    mode: Literal['bm25', 'dense', 'hybrid'] = 'hybrid'
    rerank: bool = True
    generate_answer: bool = True
    graph_enabled: bool = False
    graph_dimension: Literal['auto', 'topic', 'method', 'experiment', 'conclusion'] = 'auto'


@app.post('/api/ask')
def ask_question(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(422, '请输入问题。')
    library = {p['paper_id']: p for p in papers()}
    for identifier in request.paper_ids:
        if identifier not in library:
            raise HTTPException(404, '选择的论文不存在。')
        if not library[identifier]['indexed']:
            raise HTTPException(422, '选择的论文尚未建立索引。')
    try:
        result = rag_service.ask(question, request.paper_ids, request.top_k,
                                 request.mode, request.rerank, request.generate_answer,
                                 **({'graph_enabled': True, 'graph_dimension': request.graph_dimension}
                                    if request.graph_enabled else {}))
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    except Exception as error:
        raise HTTPException(503, 'RAG 服务暂不可用，请检查 Qdrant 和本地模型。') from error
    for hit in result['evidence']:
        paper = library[hit['paper_id']]
        hit['title'] = paper['title']
        hit['pdf_url'] = public_paper(paper)['pdf_url'] + '#page=' + str(hit['pdf_page'])
    return result


@app.get('/')
def home():
    return FileResponse(ROOT / 'frontend/index.html')


@app.get('/study')
def study():
    return FileResponse(ROOT / 'frontend/study.html')


@app.get('/graph')
def graph():
    return FileResponse(ROOT / 'frontend/graph.html')


@app.get('/review')
def review():
    return FileResponse(ROOT / 'frontend/review.html')


from backend.study import create_study_router
from backend.paper_graph import create_paper_graph_router
from backend.reviews import create_reviews_router
from backend.related_reading import create_related_reading_router
app.include_router(create_paper_graph_router(ROOT))
app.include_router(create_reviews_router(ROOT, rag_service.get_pipeline))
app.include_router(create_related_reading_router(ROOT, rag_service.get_pipeline))


@app.get('/reading')
def related_reading():
    return FileResponse(ROOT / 'frontend/related-reading.html')
app.include_router(create_study_router(ROOT, lambda identifier: next(
    (paper for paper in papers(False) if paper['paper_id'] == identifier), None)))


app.mount('/static', StaticFiles(directory=ROOT / 'frontend'), name='static')
