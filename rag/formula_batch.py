"""Checkpointed full-paper formula extraction; the extractor validates cached data."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import portalocker
import pymupdf

from rag.config import ROOT
from rag.formula_extraction import VERSION, extract_document
from rag.formula_store import publish_document
from rag.storage import atomic_json, read_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def select_papers(catalog, paper_ids=None, limit=0):
    if limit < 0:
        raise ValueError('limit must be nonnegative; 0 means all papers')
    ids = [p['paper_id'] for p in catalog]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate paper IDs in the catalog; resolve them before extraction')
    by_id = dict(zip(ids, catalog, strict=True))
    if paper_ids:
        requested = list(dict.fromkeys(paper_ids))
        unknown = set(requested)-set(by_id)
        if unknown:
            raise ValueError('Unknown downloaded paper IDs: '+', '.join(sorted(unknown)))
        papers = [by_id[pid] for pid in requested]
    else:
        papers = list(catalog)
    return papers[:limit] if limit else papers


def batch_state_path(root, state_path=None):
    root = Path(root).resolve()
    path = Path(state_path) if state_path else root/'runtime/formula_batch/progress.json'
    if not path.is_absolute():
        path = root/path
    path = path.resolve()
    if not path.is_relative_to((root/'runtime/formula_batch').resolve()) or path.suffix != '.json':
        raise ValueError('Batch state must be a JSON file under runtime/formula_batch')
    return path


def run_formula_batch(papers, *, root=ROOT, dpi=180, retry_failed=False,
                      state_path=None, emit=None):
    """Each paper is published only after a complete validated extraction.

    Re-running validates extraction caches rather than trusting a success flag.
    A changed PDF/profile or corrupt crop is therefore processed again.
    """
    root = Path(root).resolve()
    if not 72 <= dpi <= 400:
        raise ValueError('DPI must be between 72 and 400')
    papers = select_papers(papers)
    path = batch_state_path(root, state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(path)+'.lock', timeout=0):
        state = read_json(path, {'schema_version': 1, 'papers': {}})
        if state.get('schema_version') != 1 or not isinstance(state.get('papers'), dict):
            raise ValueError('Invalid batch checkpoint schema')
        if retry_failed:
            papers = [p for p in papers if state['papers'].get(p['paper_id'], {}).get('status') == 'failed']
        profile = {'extractor_version': VERSION, 'dpi': dpi, 'pymupdf': pymupdf.VersionBind}
        started = perf_counter()
        run = {'status': 'running', 'started_at': timestamp(), 'profile': profile,
               'requested_ids': [p['paper_id'] for p in papers], 'total': len(papers),
               'processed': 0, 'completed': 0, 'cached': 0, 'failed': 0,
               'formula_candidates': 0, 'numbered_candidates': 0, 'pages': 0,
               'papers_needing_layout_ocr': [], 'unresolved_regions': 0,
               'failed_ids': [], 'elapsed_seconds': 0, 'retry_failed_only': retry_failed}
        state['last_run'] = run
        atomic_json(path, state)

        def log(message):
            if emit:
                emit(message)

        def checkpoint():
            run['elapsed_seconds'] = round(perf_counter()-started, 2)
            run['updated_at'] = timestamp()
            atomic_json(path, state)

        log(f'全论文公式抽取：{len(papers)} 篇；DPI={dpi}；版本={VERSION}')
        log(f'逐篇保存进度：{path}')
        for position, paper in enumerate(papers, 1):
            pid = paper['paper_id']
            prefix = f'[{position}/{len(papers)}] {pid}'
            attempt_started = perf_counter()
            record = {'paper_id': pid, 'title': paper.get('title', ''), 'status': 'running',
                      'started_at': timestamp(), 'profile': profile}
            state['papers'][pid] = record
            checkpoint()
            log(prefix+' 开始')
            last_report = attempt_started

            def page_progress(phase, completed, total):
                nonlocal last_report
                now = perf_counter()
                if now-last_report >= 20:
                    name = {'detect': '检测', 'render': '裁剪', 'cache_check': '缓存校验', 'cache_hit': '缓存已复用'}[phase]
                    log(f'{prefix} {name} {completed}/{total} 页')
                    last_report = now

            try:
                source = (root/paper['pdf_path']).resolve()
                if not source.is_relative_to(root) or not source.is_file():
                    raise ValueError('PDF missing or outside the project: '+str(paper['pdf_path']))
                document = {**paper, '_path': source,
                            'landing_url': paper.get('landing_url') or paper.get('source_url') or ''}
                output, elements, manifest = extract_document(document, root=root, dpi=dpi, progress=page_progress)
                published = publish_document(output, root)
                record.update(status='completed', finished_at=timestamp(),
                              elapsed_seconds=round(perf_counter()-attempt_started, 2),
                              document_sha256=manifest['document_sha256'], directory=published['directory'],
                              candidates=len(elements), numbered=manifest['numbered'], pages=manifest['page_count'],
                              cache_hit=manifest['cache_hit'], pending_layout_count=len(manifest['pending_layout']),
                              pages_needing_layout_ocr=manifest['pages_needing_ocr_layout'])
                run['completed'] += 1
                run['cached'] += int(manifest['cache_hit'])
                run['formula_candidates'] += len(elements)
                run['numbered_candidates'] += manifest['numbered']
                run['pages'] += manifest['page_count']
                run['unresolved_regions'] += len(manifest['pending_layout'])
                if manifest['pages_needing_ocr_layout']:
                    run['papers_needing_layout_ocr'].append(pid)
                log(f"{prefix} 完成{'（缓存）' if manifest['cache_hit'] else ''}：{len(elements)} 个候选，"
                    f"编号 {manifest['numbered']}，待处理区域 {len(manifest['pending_layout'])}，"
                    f"无文字层页 {len(manifest['pages_needing_ocr_layout'])}")
            except KeyboardInterrupt:
                record.update(status='interrupted', finished_at=timestamp())
                run.update(status='interrupted', interrupted_paper_id=pid, finished_at=timestamp())
                checkpoint()
                log(prefix+' 已中断；重新运行同一入口即可续跑。')
                return {**run, 'state_path': str(path)}
            except Exception as error:
                record.update(status='failed', finished_at=timestamp(), error_type=type(error).__name__,
                              error_message=str(error).replace('\n', ' ')[:600],
                              elapsed_seconds=round(perf_counter()-attempt_started, 2))
                run['failed'] += 1
                run['failed_ids'].append(pid)
                log(f"{prefix} 失败：{record['error_type']}: {record['error_message']}；继续下一篇。")
            run['processed'] += 1
            checkpoint()
        run.update(status='completed_with_failures' if run['failed'] else 'completed', finished_at=timestamp())
        checkpoint()
        return {**run, 'state_path': str(path)}


def catalog_counts(catalog):
    return dict(Counter('uploaded' if p['paper_id'].startswith('upload-') else 'curated' for p in catalog))
