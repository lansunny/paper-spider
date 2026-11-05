"""Versioned, local paper reading state, annotations and annotated PDF export."""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
import pymupdf


COLORS = ('#f6d96b', '#74a9f5', '#f29baf', '#8cd3af', '#345a89')


def now():
    return datetime.now(timezone.utc).isoformat()


class Annotation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: UUID
    kind: Literal['highlight', 'underline', 'ink', 'rectangle', 'note']
    page: int = Field(ge=1)
    color: Literal['#f6d96b', '#74a9f5', '#f29baf', '#8cd3af', '#345a89'] = '#f6d96b'
    rects: list[list[float]] = Field(default_factory=list, max_length=200)
    points: list[list[float]] = Field(default_factory=list, max_length=2500)
    width: float = Field(default=2, ge=.5, le=12)
    quote: str = Field(default='', max_length=15000)
    text: str = Field(default='', max_length=10000)

    @model_validator(mode='after')
    def geometry(self):
        for rect in self.rects:
            if (len(rect) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in rect)
                    or rect[0] >= rect[2] or rect[1] >= rect[3]):
                raise ValueError('标注矩形必须位于页面内，并具有有效面积。')
        for point in self.points:
            if len(point) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in point):
                raise ValueError('画笔坐标必须位于页面内。')
        if self.kind == 'ink':
            if len(self.points) < 2 or self.rects:
                raise ValueError('画笔需要至少两个坐标点。')
        elif not self.rects or self.points:
            raise ValueError('此标注需要页面矩形坐标。')
        return self


class AnnotationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    annotation: Annotation


class StatePatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    page: int | None = Field(default=None, ge=1)
    scroll_position: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    zoom: str | None = Field(default=None, pattern=r'^(fit|(?:50|75|100|125|150|175|200))$')
    notes: str | None = Field(default=None, max_length=100000)
    notes_revision: int | None = Field(default=None, ge=0)
    bookmarks: list[int] | None = Field(default=None, max_length=1000)
    reading_status: Literal['reading', 'finished'] | None = None

    @model_validator(mode='after')
    def notes_have_revision(self):
        if self.notes is not None and self.notes_revision is None:
            raise ValueError('保存笔记需要携带当前版本号。')
        return self


class NotesConflict(Exception):
    def __init__(self, notes, revision):
        self.notes, self.revision = notes, revision


class StudyStore:
    def __init__(self, root):
        self.path = Path(root) / 'runtime/study/study.sqlite3'

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA foreign_keys=ON')
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS reading_state (
                    paper_id TEXT NOT NULL, document_sha256 TEXT NOT NULL,
                    page INTEGER NOT NULL DEFAULT 1, zoom TEXT NOT NULL DEFAULT 'fit',
                    scroll_position REAL NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT '', notes_revision INTEGER NOT NULL DEFAULT 0,
                    bookmarks TEXT NOT NULL DEFAULT '[]', reading_status TEXT NOT NULL DEFAULT 'reading',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (paper_id, document_sha256)
                );
                CREATE TABLE IF NOT EXISTS annotations (
                    paper_id TEXT NOT NULL, document_sha256 TEXT NOT NULL, id TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (paper_id, document_sha256, id),
                    FOREIGN KEY (paper_id, document_sha256)
                      REFERENCES reading_state(paper_id, document_sha256)
                );
            ''')
            columns = {row[1] for row in connection.execute('PRAGMA table_info(reading_state)')}
            if 'scroll_position' not in columns:
                connection.execute('ALTER TABLE reading_state ADD COLUMN scroll_position REAL NOT NULL DEFAULT 0')
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def ensure(db, paper_id, digest):
        db.execute('INSERT OR IGNORE INTO reading_state(paper_id,document_sha256,updated_at) VALUES(?,?,?)',
                   (paper_id, digest, now()))

    def read(self, paper_id, digest):
        with self.connection() as db:
            self.ensure(db, paper_id, digest)
            row = dict(db.execute('SELECT * FROM reading_state WHERE paper_id=? AND document_sha256=?',
                                  (paper_id, digest)).fetchone())
            row['bookmarks'] = json.loads(row['bookmarks'])
            row['annotations'] = [dict(json.loads(a['payload']), created_at=a['created_at'], updated_at=a['updated_at'])
                                  for a in db.execute('SELECT * FROM annotations WHERE paper_id=? AND document_sha256=? ORDER BY created_at,id',
                                                      (paper_id, digest))]
            return row

    def recent(self, limit=24):
        with self.connection() as db:
            return [dict(row) for row in db.execute('''
                SELECT s.paper_id,s.document_sha256,s.page,s.reading_status,s.updated_at,
                       length(s.notes) AS notes_characters,
                       (SELECT count(*) FROM annotations a WHERE a.paper_id=s.paper_id
                        AND a.document_sha256=s.document_sha256) AS annotation_count
                FROM reading_state s ORDER BY s.updated_at DESC LIMIT ?''', (limit,))]

    def update(self, paper_id, digest, patch, pages):
        if patch.page is not None and patch.page > pages:
            raise ValueError('页码超出范围。')
        if patch.bookmarks is not None and any(p < 1 or p > pages for p in patch.bookmarks):
            raise ValueError('书签页码超出范围。')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self.ensure(db, paper_id, digest)
            current = db.execute('SELECT notes,notes_revision FROM reading_state WHERE paper_id=? AND document_sha256=?',
                                 (paper_id, digest)).fetchone()
            if patch.notes is not None and patch.notes_revision != current['notes_revision']:
                raise NotesConflict(current['notes'], current['notes_revision'])
            changes = {'updated_at': now()}
            for key in ('page', 'zoom', 'scroll_position', 'notes', 'reading_status'):
                value = getattr(patch, key)
                if value is not None:
                    changes[key] = value
            if patch.notes is not None:
                changes['notes_revision'] = current['notes_revision'] + 1
            if patch.bookmarks is not None:
                changes['bookmarks'] = json.dumps(sorted(set(patch.bookmarks)))
            sql = ','.join(key + '=?' for key in changes)
            db.execute('UPDATE reading_state SET ' + sql + ' WHERE paper_id=? AND document_sha256=?',
                       (*changes.values(), paper_id, digest))
        return self.read(paper_id, digest)

    def put_annotation(self, paper_id, digest, annotation, pages):
        if annotation.page > pages:
            raise ValueError('标注页码超出范围。')
        payload = annotation.model_dump(mode='json')
        stamp = now()
        with self.connection() as db:
            self.ensure(db, paper_id, digest)
            db.execute('''INSERT INTO annotations VALUES(?,?,?,?,?,?)
                ON CONFLICT(paper_id,document_sha256,id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at''',
                       (paper_id, digest, str(annotation.id), json.dumps(payload, ensure_ascii=False), stamp, stamp))
            row = db.execute('SELECT created_at FROM annotations WHERE paper_id=? AND document_sha256=? AND id=?',
                             (paper_id, digest, str(annotation.id))).fetchone()
        return {**payload, 'created_at': row['created_at'], 'updated_at': stamp}

    def delete_annotation(self, paper_id, digest, identifier):
        with self.connection() as db:
            cursor = db.execute('DELETE FROM annotations WHERE paper_id=? AND document_sha256=? AND id=?',
                                (paper_id, digest, str(identifier)))
        return bool(cursor.rowcount)


@lru_cache(maxsize=32)
def document_info(path, mtime_ns, size):
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pymupdf.open(source) as pdf:
        outline = [{'level': level, 'title': title, 'page': page} for level, title, page in pdf.get_toc()
                   if 1 <= page <= len(pdf)][:1000]
        origin = 'pdf'
        if not outline:
            origin = 'detected'
            seen = set()
            pattern = re.compile(r'^(?:\d{1,2}(?:\.\d{1,2}){0,3}\s+[A-Z][A-Za-z]|Abstract$|References$|Acknowledg|Appendix\b)', re.I)
            for index, page in enumerate(pdf):
                for block in page.get_text('blocks'):
                    lines = [line.strip() for line in block[4].splitlines() if line.strip()]
                    first = lines[0] if lines else ''
                    if len(lines) > 1 and re.fullmatch(r'\d{1,2}(?:\.\d{1,2}){0,3}', first):
                        first += ' ' + lines[1]
                    if 3 < len(first) < 100 and pattern.match(first) and first not in seen:
                        seen.add(first)
                        number = first.split()[0]
                        outline.append({'level': min(3, number.count('.') + 1), 'title': first, 'page': index + 1})
        return {'document_sha256': digest, 'pages': len(pdf), 'outline': outline,
                'outline_source': origin, 'pdf_title': pdf.metadata.get('title', ''),
                'pdf_author': pdf.metadata.get('author', '')}


def annotated_pdf(path, annotations):
    with pymupdf.open(path) as pdf:
        for item in annotations:
            a = Annotation.model_validate({k: v for k, v in item.items() if k not in {'created_at', 'updated_at'}})
            page = pdf[a.page - 1]
            width, height = page.rect.width, page.rect.height
            def point(p):
                return pymupdf.Point(p[0] * width, p[1] * height) * page.derotation_matrix
            rects = [pymupdf.Rect(r[0] * width, r[1] * height, r[2] * width, r[3] * height)
                     * page.derotation_matrix for r in a.rects]
            rgb = tuple(int(a.color[i:i+2], 16) / 255 for i in (1, 3, 5))
            if a.kind in {'highlight', 'underline'}:
                quads = []
                for r in a.rects:
                    quads.append(pymupdf.Quad(point([r[0], r[1]]), point([r[2], r[1]]),
                                             point([r[0], r[3]]), point([r[2], r[3]])))
                mark = page.add_highlight_annot(quads) if a.kind == 'highlight' else page.add_underline_annot(quads)
            elif a.kind == 'ink':
                mark = page.add_ink_annot([[tuple(point(p)) for p in a.points]])
                mark.set_border(width=a.width)
            elif a.kind == 'rectangle':
                mark = page.add_rect_annot(rects[0])
                mark.set_border(width=a.width)
            else:
                mark = page.add_text_annot(rects[0].tl, a.text or '笔记')
            mark.set_colors(stroke=rgb)
            mark.set_info(title='论文蛛 · Paper Spider', content=a.text or a.quote)
            mark.update(opacity=.35 if a.kind == 'highlight' else 1)
        return pdf.tobytes(garbage=3, deflate=True)


def create_study_router(root, resolve_paper):
    root = Path(root).resolve()
    store = StudyStore(root)
    router = APIRouter(prefix='/api/study/papers', tags=['Paper study'])

    def document(paper_id, digest=None):
        paper = resolve_paper(paper_id)
        if not paper:
            raise HTTPException(404, '论文不存在。')
        path = (root / paper['pdf_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, 'PDF 文件不存在。')
        stat = path.stat()
        try:
            info = document_info(str(path), stat.st_mtime_ns, stat.st_size)
        except (ValueError, RuntimeError):
            raise HTTPException(422, '无法读取 PDF。') from None
        if digest is not None and digest != info['document_sha256']:
            raise HTTPException(409, '论文文件版本已变化，请重新打开论文。')
        return paper, path, info

    @router.get('/recent')
    def recent_papers():
        result = []
        seen = set()
        for record in store.recent():
            if record['paper_id'] in seen:
                continue
            try:
                paper, _, info = document(record['paper_id'], record['document_sha256'])
            except HTTPException:
                continue
            seen.add(record['paper_id'])
            result.append({**record, 'title': paper['title'], 'year': paper.get('year'), 'pages': info['pages']})
            if len(result) == 6:
                break
        return {'papers': result}

    @router.get('/{paper_id}')
    def metadata(paper_id: str):
        paper, _, info = document(paper_id)
        return {'paper': {k: v for k, v in paper.items() if k not in {'pdf_path', 'sha256'}},
                **info, 'state': store.read(paper_id, info['document_sha256'])}

    @router.patch('/{paper_id}/state')
    def update_state(paper_id: str, patch: StatePatch):
        _, _, info = document(paper_id, patch.document_sha256)
        try:
            return store.update(paper_id, patch.document_sha256, patch, info['pages'])
        except NotesConflict as error:
            raise HTTPException(409, {'message': '另一窗口已更新笔记，本地草稿已保留。',
                                      'notes': error.notes, 'notes_revision': error.revision}) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @router.put('/{paper_id}/annotations/{identifier}')
    def put_annotation(paper_id: str, identifier: UUID, request: AnnotationRequest):
        _, _, info = document(paper_id, request.document_sha256)
        if identifier != request.annotation.id:
            raise HTTPException(422, '标注编号不一致。')
        try:
            return store.put_annotation(paper_id, request.document_sha256, request.annotation, info['pages'])
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @router.delete('/{paper_id}/annotations/{identifier}')
    def delete_annotation(paper_id: str, identifier: UUID, document_sha256: str):
        document(paper_id, document_sha256)
        return {'deleted': store.delete_annotation(paper_id, document_sha256, identifier)}

    @router.get('/{paper_id}/export')
    def export(paper_id: str, document_sha256: str, format: Literal['pdf', 'json', 'md'] = 'pdf'):
        paper, path, info = document(paper_id, document_sha256)
        state = store.read(paper_id, document_sha256)
        filename = 'paper-spider-' + re.sub(r'[^\w.-]', '_', paper_id) + '.' + format
        if format == 'pdf':
            content = annotated_pdf(path, state['annotations'])
            media = 'application/pdf'
        elif format == 'json':
            content = json.dumps({'schema_version': 1, 'title': paper['title'], **state}, ensure_ascii=False, indent=2)
            media = 'application/json'
        else:
            lines = ['# ' + paper['title'], '', '## 阅读笔记', '', state['notes'], '', '## 原文标注', '']
            for a in state['annotations']:
                lines.extend(['### PDF 第 ' + str(a['page']) + ' 页 · ' + a['kind'], '',
                              '\n'.join('> ' + line for line in a['quote'].splitlines()), '', a['text'], ''])
            content = '\n'.join(lines)
            media = 'text/markdown'
        return Response(content, media_type=media, headers={
            'Content-Disposition': 'attachment; filename="' + filename + '"', 'Cache-Control': 'no-store'})

    return router
