"""Local CLIP features for immutable Docling PDF crops."""
from contextlib import closing
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps
import portalocker

from rag.config import ROOT
from rag.storage import atomic_json, read_json
from scripts.docling_prepare import WORK, digest

CLIP_MODEL = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
CLIP_REVISION = '1a25a446712ba5ee05982a381eed697ef9b435cf'
CLIP_PATH = ROOT/'runtime/models'/('clip-' + CLIP_REVISION[:12])
DIM = 512
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def image_rgb(blob):
    if not 0 < len(blob) <= MAX_IMAGE_BYTES:
        raise ValueError('Image must be at most 8 MiB')
    try:
        with Image.open(io.BytesIO(blob), formats=['PNG', 'JPEG', 'WEBP']) as image:
            if image.width * image.height > 20_000_000 or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Supply one image with at most 20 million pixels')
            return ImageOps.exif_transpose(image).convert('RGB')
    except (OSError, Image.DecompressionBombError, SyntaxError) as error:
        raise ValueError('Supply a valid PNG, JPEG, or WebP image') from error


class LocalClip:
    def __init__(self, device='cpu'):
        self.device = device
        self.lock = RLock()
        self._model = self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            if not (CLIP_PATH/'model.safetensors').is_file():
                raise RuntimeError('Local CLIP assets are missing')
            self._processor = CLIPProcessor.from_pretrained(CLIP_PATH, local_files_only=True, use_fast=False)
            self._model = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True, use_safetensors=True).to(self.device).eval()

    def _encode(self, *, images=None, text=None):
        import torch

        with self.lock, torch.inference_mode():
            self._load()
            if images is not None:
                # Keep corners and axis labels: contain the whole crop instead of center-cropping it.
                size = self._model.config.vision_config.image_size
                images = [ImageOps.pad(picture, (size, size), color='white') for picture in images]
                batch = self._processor(images=images, return_tensors='pt').to(self.device)
                features = self._model.get_image_features(**batch)
            else:
                batch = self._processor(text=[text], return_tensors='pt', truncation=True, max_length=77).to(self.device)
                features = self._model.get_text_features(**batch)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            return torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()

    def text(self, text):
        return self._encode(text=text)[0]

    def images(self, images):
        return self._encode(images=images)


def visual_directory(profile, pilot):
    return WORK/profile/('clip-pilot' if pilot else 'clip')


def build_visual(profile, manifest, *, pilot, encoder):
    """Publish a sidecar only after every included crop has been verified and encoded."""
    directory = visual_directory(profile, pilot)
    directory.mkdir(parents=True, exist_ok=True)
    db_path = ROOT/manifest['object_db']
    object_snapshot = manifest['object_snapshot_id']
    with portalocker.Lock(str(directory/'build.lock'), timeout=0):
        with closing(sqlite3.connect(db_path.as_uri()+'?mode=ro', uri=True)) as db:
            entries = []
            for folder, payload in db.execute("SELECT artifact_dir,payload FROM objects WHERE kind='figure' ORDER BY object_id"):
                obj = json.loads(payload)
                if manifest['documents'].get(obj['paper_id']) != obj['pdf_sha256']:
                    continue
                for crop in obj['crops']:
                    base = (ROOT/folder).resolve()
                    path = (base/crop['path']).resolve()
                    if not path.is_relative_to(base):
                        raise ValueError('Figure crop escaped its artifact directory')
                    entries.append((obj['object_id'], obj['paper_id'], str(path), crop['sha256']))
        if not entries:
            raise ValueError('No figure crops in the requested snapshot')
        vectors = []
        for offset in range(0, len(entries), 16):
            images = []
            for _, _, path, expected in entries[offset:offset+16]:
                blob = Path(path).read_bytes()
                if sha256(blob).hexdigest() != expected:
                    raise ValueError('Figure crop changed since object materialization')
                images.append(image_rgb(blob))
            batch = encoder.images(images)
            if batch.shape != (len(images), DIM) or not np.isfinite(batch).all() or not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=.002):
                raise ValueError('Invalid CLIP vectors')
            vectors.append(batch)
            if offset % 256 == 0 or offset+16 >= len(entries):
                print(f'CLIP crops {min(offset+16,len(entries))}/{len(entries)}', flush=True)
        target = directory/(uuid4().hex+'.npz')
        temp = target.with_suffix('.tmp')
        try:
            with temp.open('wb') as stream:
                np.savez(stream, vectors=np.concatenate(vectors), ids=[e[0] for e in entries], papers=[e[1] for e in entries])
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        result = {'profile': profile, 'object_snapshot': object_snapshot, 'index_snapshot': manifest['snapshot_id'],
                  'model': CLIP_MODEL, 'revision': CLIP_REVISION, 'preprocessing': 'white-pad-v1',
                  'crops': len(entries), 'figures': len({e[0] for e in entries}), 'file': target.name, 'sha256': digest(target)}
        atomic_json(directory/'CURRENT.json', result)
        return result


class FigureVectors:
    def __init__(self, profile, manifest, pilot):
        directory = visual_directory(profile, pilot)
        marker = read_json(directory/'CURRENT.json')
        if not marker or any(marker.get(k) != v for k, v in {
                'profile': profile, 'index_snapshot': manifest['snapshot_id'], 'revision': CLIP_REVISION,
                'object_snapshot': manifest['object_snapshot_id'], 'preprocessing': 'white-pad-v1'}.items()):
            raise RuntimeError('CLIP vectors have not been built for this object snapshot')
        path = (directory/marker['file']).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != marker['sha256']:
            raise ValueError('CLIP artifact checksum mismatch')
        with np.load(path, allow_pickle=False) as stored:
            self.vectors, self.ids, self.papers = (stored[k] for k in ('vectors', 'ids', 'papers'))
        count = marker['crops']
        if (self.vectors.shape != (count, DIM) or self.ids.shape != (count,) or self.papers.shape != (count,)
                or not np.isfinite(self.vectors).all() or not set(self.papers).issubset(manifest['documents'])):
            raise ValueError('CLIP index is inconsistent with the object snapshot')

    def rank(self, vector, limit, paper_id=None, *, paper_ids=None):
        scores = self.vectors @ vector
        scope = set(paper_ids or ([paper_id] if paper_id else []))
        best = {}
        for obj, paper, score in zip(self.ids, self.papers, scores, strict=True):
            if not scope or paper in scope:
                best[str(obj)] = max(float(score), best.get(str(obj), -float('inf')))
        return [{'object_id': obj, 'score': best[obj]}
                for obj in sorted(best, key=lambda obj: (-best[obj], obj))[:limit]]
