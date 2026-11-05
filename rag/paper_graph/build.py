"""Pilot preparation, cached paper profiles, encoding and immutable graph bundles."""
import importlib.metadata
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from rag.formula_store import digest_file
from rag.ingestion import fingerprint
from rag.storage import atomic_json, atomic_text, read_json
from .extraction import PROMPT_VERSION, extract_paper
from .graph import similarity_graph
from .schema import DIMENSIONS, LABELS, SOURCE_VERSION, StrictModel, validate_description
from .sources import load_sources, select_pilot, source_batches
from .store import publish, write_database


class GraphSettings(StrictModel):
    schema_version: int = 1
    limit: int = Field(default=50, ge=1, le=500)
    seed: int = Field(default=42, ge=0)
    workers: int = Field(default=2, ge=1, le=4)
    max_source_chars_per_call: int = Field(default=60000, ge=4000, le=120000)
    max_output_tokens: int = Field(default=6500, ge=2500, le=16000)
    neighbors: int = Field(default=10, ge=1, le=50)
    thresholds: dict[str, float]
    resolution: float = Field(default=1, gt=0)
    threshold_status: str = 'initial_uncalibrated'
    partition: str = 'RBConfigurationVertexPartition'
    symmetrization: str = 'union'

    @model_validator(mode='after')
    def supported_graph(self):
        if (set(self.thresholds) != set(DIMENSIONS) or any(not 0 <= t < 1 for t in self.thresholds.values())
                or self.partition != 'RBConfigurationVertexPartition' or self.symmetrization != 'union'
                or self.schema_version != 1):
            raise ValueError('Unsupported graph schema, thresholds or community method')
        return self


def prepare(root, papers, settings, generation):
    records = []
    for paper in select_pilot(papers, settings.limit, settings.seed):
        sources, origin = load_sources(paper, root)
        sources, batches = source_batches(sources, settings.max_source_chars_per_call)
        identity = fingerprint({'document': paper['sha256'], 'title': paper['title'],
            'source_ids': [s['source_id'] for s in sources], 'source_version': SOURCE_VERSION,
            'prompt_version': PROMPT_VERSION, 'model': generation.model, 'endpoint': generation.base_url,
            'max_output_tokens': settings.max_output_tokens,
            'batch_sources': [[s['source_id'] for s in batch] for batch in batches]})
        records.append({'paper': paper, 'sources': sources, 'batches': batches, 'origin': origin,
                        'cache_key': identity})
    return records


def profile_cache(root, record):
    return root / 'runtime/paper_graph/cache/profiles' / (record['cache_key'] + '.json')


def read_profile(root, record):
    value = read_json(profile_cache(root, record))
    if not value:
        return None
    if (value['cache_key'] != record['cache_key'] or value['document_sha256'] != record['paper']['sha256']
            or fingerprint(value['dimensions']) != value['profile_sha256']):
        raise ValueError('Cached paper profile integrity mismatch')
    validate_description(value['dimensions'], {s['source_id'] for s in record['sources']})
    return value


def make_profile(root, record, generation, token_count, max_length):
    cached = read_profile(root, record)
    if cached:
        return cached, True
    dimensions, steps = extract_paper(record['paper'], record['batches'], generation,
        root / 'runtime/paper_graph/cache/requests', token_count, max_length)
    if any(token_count(d['description']) > max_length for d in dimensions.values() if d['status'] == 'available'):
        raise ValueError('Paper profile exceeds the embedding input length')
    value = {'cache_key': record['cache_key'], 'paper_id': record['paper']['paper_id'],
        'document_sha256': record['paper']['sha256'], 'source_version': SOURCE_VERSION,
        'source_origin': record['origin'], 'prompt_version': PROMPT_VERSION,
        'model': generation.model, 'profile_sha256': fingerprint(dimensions), 'dimensions': dimensions,
        'requests': [{'request_sha256': step['request_sha256'], 'attempts': step['attempts']} for step in steps],
        'validation': 'schema_source_ids_and_exact_source_offsets_only_not_entailment',
        'created_at': datetime.now(timezone.utc).isoformat()}
    atomic_json(profile_cache(root, record), value)
    return value, False


def encode_descriptions(texts, models, settings, root):
    encoder_profile = {k: settings.index_profile()[k] for k in (
        'embedding_model', 'embedding_revision', 'vector_size', 'embedding_max_length', 'pooling', 'normalized')}
    directory = root / 'runtime/paper_graph/cache/vectors' / fingerprint(encoder_profile)
    directory.mkdir(parents=True, exist_ok=True)
    output, missing = [None] * len(texts), []
    for index, text in enumerate(texts):
        path = directory / (fingerprint(text) + '.npy')
        try:
            value = np.load(path, allow_pickle=False)
            if (value.shape != (settings.vector_size,) or not np.isfinite(value).all()
                    or not np.isclose(np.linalg.norm(value), 1, atol=.01)):
                raise ValueError('Invalid vector cache')
            output[index] = value
        except (OSError, ValueError):
            missing.append(index)
    if missing:
        values = models.encode([texts[i] for i in missing])
        for index, value in zip(missing, values, strict=True):
            output[index] = value
            descriptor, temporary = tempfile.mkstemp(dir=directory, suffix='.npy')
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    np.save(stream, value, allow_pickle=False)
                os.replace(temporary, directory / (fingerprint(texts[index]) + '.npy'))
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    return np.asarray(output, dtype=np.float32).reshape(len(texts), settings.vector_size), encoder_profile


def build_bundle(root, records, graph_settings, embedding_settings, models, source_snapshot_id):
    profiles = {r['paper']['paper_id']: read_profile(root, r) for r in records}
    if any(p is None for p in profiles.values()):
        raise ValueError('All selected papers must finish extraction before graph publication')
    for record in records:
        path = (root / record['paper']['pdf_path']).resolve()
        if not path.is_relative_to(root.resolve()) or digest_file(path) != record['paper']['sha256']:
            raise ValueError('A source PDF changed during extraction; graph publication stopped')
    build_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + fingerprint([
        graph_settings.model_dump(), {p: v['profile_sha256'] for p, v in profiles.items()}])[:10]
    directory = root / 'runtime/paper_graph/builds' / build_id
    directory.mkdir(parents=True, exist_ok=False)
    views, arrays, encoder_profile = {}, {}, None
    for dimension in DIMENSIONS:
        paper_ids = sorted(p for p, profile in profiles.items() if profile['dimensions'][dimension]['status'] == 'available')
        texts = [profiles[p]['dimensions'][dimension]['description'] for p in paper_ids]
        vectors, encoder_profile = encode_descriptions(texts, models, embedding_settings, root)
        arrays[dimension + '_paper_ids'] = np.asarray(paper_ids, dtype=str)
        arrays[dimension + '_vectors'] = vectors
        views[dimension] = similarity_graph(paper_ids, vectors, dimension=dimension,
            neighbors=graph_settings.neighbors, threshold=graph_settings.thresholds[dimension],
            resolution=graph_settings.resolution, seed=graph_settings.seed)
        for node in views[dimension]['nodes']:
            node['profile_sha256'] = profiles[node['paper_id']]['profile_sha256']
        print(f"{LABELS[dimension]}: {len(paper_ids)} papers, {len(views[dimension]['edges'])} edges, "
              f"{len(views[dimension]['communities'])} communities", flush=True)
    np.savez_compressed(directory / 'vectors.npz', **arrays)
    atomic_json(directory / 'views.json', views)
    atomic_text(directory / 'profiles.jsonl', ''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in profiles.values()))
    write_database(directory / 'graph.sqlite3', [r['paper'] for r in records], profiles,
                   {r['paper']['paper_id']: r['sources'] for r in records}, views)
    request_records = {step['request_sha256']: step for profile in profiles.values() for step in profile['requests']}
    usage = {key: sum(attempt['usage'].get(key, 0) for step in request_records.values() for attempt in step['attempts'])
             for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    manifest = {'schema_version': 1, 'status': 'completed', 'build_id': build_id,
        'created_at': datetime.now(timezone.utc).isoformat(), 'papers': len(records),
        'paper_ids': list(profiles), 'source_snapshot_id': source_snapshot_id,
        'settings': graph_settings.model_dump(), 'encoder': encoder_profile,
        'prompt_version': PROMPT_VERSION, 'source_version': SOURCE_VERSION,
        'selection_method': 'seeded_title_bucket_pilot_with_anchor_papers',
        'profile_hashes': {p: value['profile_sha256'] for p, value in profiles.items()},
        'extraction_model': sorted({p['model'] for p in profiles.values()}),
        'dimensions': {d: views[d]['statistics'] for d in DIMENSIONS},
        'dependencies': {p: importlib.metadata.version(p) for p in ('igraph', 'leidenalg', 'numpy')},
        'extraction_usage_including_reused_steps': usage,
        'unique_model_steps': len(request_records),
        'validation': 'pipeline_smoke_run_not_quality_benchmark',
        'edge_meaning': 'per_dimension_semantic_similarity_only',
        'retrieval_integration': 'not_enabled',
        'artifacts': {name: digest_file(directory / name) for name in (
            'vectors.npz', 'views.json', 'profiles.jsonl', 'graph.sqlite3')}}
    publish(root, directory, manifest)
    return directory, manifest
