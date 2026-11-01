"""Pinned public model downloads with resumable ranges and repository hash checks."""
import hashlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from rag.config import ROOT
from rag.storage import atomic_json

FILES = {'config.json', 'modules.json', 'config_sentence_transformers.json',
         'sentence_bert_config.json', '1_Pooling/config.json', 'pytorch_model.bin',
         'model.safetensors', 'sentencepiece.bpe.model', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json'}


def file_matches(path, entry):
    if not path.exists() or path.stat().st_size != entry['size']:
        return False
    digest = hashlib.sha256() if 'lfs' in entry else hashlib.sha1()
    if 'lfs' not in entry:
        digest.update(f"blob {entry['size']}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    expected = entry['lfs']['sha256'] if 'lfs' in entry else entry['blobId']
    return digest.hexdigest() == expected


def download_file(client, url, path, entry, progress=print):
    if file_matches(path, entry):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.download')
    if entry['size'] < 32 * 1024 * 1024:
        response = client.get(url)
        response.raise_for_status()
        temporary.write_bytes(response.content)
    else:
        block_size = 8 * 1024 * 1024
        count = math.ceil(entry['size'] / block_size)
        pieces = path.parent / ('.parts-' + entry['lfs']['sha256'][:16])
        pieces.mkdir(exist_ok=True)

        def part(index):
            start = index * block_size
            end = min(start + block_size, entry['size']) - 1
            target = pieces / str(index)
            if target.exists() and target.stat().st_size == end - start + 1:
                return target
            for attempt in range(3):
                try:
                    response = client.get(url + f'?download=true&part={index}&attempt={attempt}',
                                          headers={'Range': f'bytes={start}-{end}'})
                    response.raise_for_status()
                    expected_range = f"bytes {start}-{end}/{entry['size']}"
                    if response.headers.get('content-range') != expected_range or len(response.content) != end - start + 1:
                        raise ValueError('Model server returned an unexpected byte range.')
                    target.write_bytes(response.content)
                    return target
                except (httpx.HTTPError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(part, index) for index in range(count)]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed % 16 == 0 or completed == count:
                    progress(f'{path.name}: {completed}/{count} parts')
        with temporary.open('wb') as output:
            for index in range(count):
                with (pieces / str(index)).open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        output.write(block)
    if not file_matches(temporary, entry):
        raise ValueError(f'Model checksum mismatch: {path.name}; download is not activated.')
    os.replace(temporary, path)
    if entry['size'] >= 32 * 1024 * 1024:
        # Remove only numbered parts created for this verified download.
        for index in range(count):
            (pieces / str(index)).unlink(missing_ok=True)
        if not any(pieces.iterdir()):
            pieces.rmdir()


def prepare_models(settings, root: Path = ROOT, endpoint=None, progress=print):
    endpoint = (endpoint or os.getenv('HF_ENDPOINT') or 'https://huggingface.co').rstrip('/')
    specs = [('embedding', settings.embedding_model, settings.embedding_revision, settings.embedding_path),
             ('reranker', settings.reranker_model, settings.reranker_revision, settings.reranker_path)]
    with httpx.Client(timeout=httpx.Timeout(120, connect=20), follow_redirects=True) as client:
        for kind, repo, revision, relative in specs:
            if not revision:
                raise ValueError('A pinned model revision is required.')
            progress(f'Preparing {repo}@{revision}')
            metadata = client.get(f'{endpoint}/api/models/{repo}/revision/{revision}?blobs=true')
            metadata.raise_for_status()
            catalog = metadata.json()
            if catalog['sha'] != revision:
                raise ValueError('Model repository revision mismatch.')
            files = [item for item in catalog['siblings'] if item['rfilename'] in FILES]
            if any(item['rfilename'] == 'model.safetensors' for item in files):
                files = [item for item in files if item['rfilename'] != 'pytorch_model.bin']
            if not any(item['rfilename'] in {'pytorch_model.bin', 'model.safetensors'} for item in files):
                raise ValueError('Pinned model repository contains no supported weights.')
            path = root / relative
            for entry in files:
                download_file(client, f"{endpoint}/{repo}/resolve/{revision}/{entry['rfilename']}",
                              path / entry['rfilename'], entry, progress)
            atomic_json(path / 'paperrag-model.json', {'model_id': repo, 'revision': revision,
                                                      'kind': kind, 'verified_files': files})
            progress(f'{kind} ready; repository hashes verified')
