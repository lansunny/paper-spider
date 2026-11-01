"""The complete retrieval -> rerank -> context -> generation pipeline."""
from threading import RLock
from time import perf_counter
from uuid import uuid4

from rag.config import ROOT
from rag.embeddings import LocalModels
from rag.generation import generate, generation_settings, select_context
from rag.ingestion import current_chunks, current_manifest
from rag.retrieval import LexicalIndex, reciprocal_rank_fusion, rerank_candidates
from rag.routing import PaperRouter
from rag.storage import atomic_json
from rag.vector_store import QdrantStore


class Pipeline:
    def __init__(self, settings, root=ROOT, models=None, store=None):
        self.settings, self.root = settings, root
        self.models = models or LocalModels(settings, root)
        self.store = store or QdrantStore(settings)
        self.lock = RLock()
        self._snapshot = self._lexical = self._router = None

    def snapshot(self):
        with self.lock:
            latest = current_manifest(self.settings, self.root)
            if latest is None or self._snapshot is None or latest['snapshot_id'] != self._snapshot['snapshot_id']:
                manifest, chunks = current_chunks(self.settings, self.root)
                self._snapshot, self._lexical = manifest, LexicalIndex(chunks)
                self._router = (PaperRouter.from_project(manifest['documents'], self.root)
                                if self.settings.corpus == 'papers' else None)
            if self._snapshot['profile'] != self.settings.index_profile():
                raise ValueError('索引配置不匹配，请重新建立独立索引。')
            return self._snapshot, self._lexical

    def retrieve(self, question, mode='hybrid', top_k=5, rerank=True, paper_ids=None, exact=False,
                 *, lexical_query=None, auto_route=True, collect_trace=False):
        if mode not in {'bm25', 'dense', 'hybrid'} or not 1 <= top_k <= 100:
            raise ValueError('Invalid retrieval mode/top_k.')
        question = question.strip()
        if not question:
            raise ValueError('请输入问题。')
        manifest, lexical = self.snapshot()
        if paper_ids and not set(paper_ids).issubset(manifest['documents']):
            raise ValueError('选择的论文尚未进入当前索引。')
        if paper_ids:
            route = {'mode': 'manual', 'paper_ids': list(paper_ids), 'reason': 'explicit_user_selection'}
        elif auto_route and self._router is not None:
            route = self._router.route(question)
            paper_ids = route['paper_ids'] or None
        else:
            route = {'mode': 'global', 'paper_ids': [], 'reason': 'routing_disabled_or_other_corpus'}
        lexical_text = question if lexical_query is None else lexical_query.strip()
        if not lexical_text:
            raise ValueError('BM25检索句不能为空。')
        counts, timings, branches = {}, {}, {}
        candidates = max(top_k, self.settings.candidate_k)
        if mode in {'bm25', 'hybrid'}:
            start = perf_counter()
            branches['bm25'] = lexical.search(lexical_text, candidates, paper_ids)
            timings['bm25_ms'] = (perf_counter() - start) * 1000
        if mode in {'dense', 'hybrid'}:
            start = perf_counter()
            self.store.validate_collection()
            vector = self.models.query_vector(question)
            timings['query_embedding_ms'] = (perf_counter() - start) * 1000
            start = perf_counter()
            branches['dense'] = self.store.search(vector, manifest['snapshot_id'], candidates, paper_ids, exact)
            timings['dense_search_ms'] = (perf_counter() - start) * 1000
        if paper_ids and any(hit['paper_id'] not in paper_ids for values in branches.values() for hit in values):
            raise ValueError('检索结果超出了选定论文范围。')
        counts.update({key: len(value) for key, value in branches.items()})
        hits = reciprocal_rank_fusion(branches, self.settings.rrf_k) if mode == 'hybrid' else branches[mode]
        counts['merged'] = len(hits)
        candidate_hits = hits[:max(top_k, self.settings.rerank_k)]
        counts['rerank_candidates'] = len(candidate_hits)
        if rerank:
            start = perf_counter()
            ranked = rerank_candidates(self.models, question, candidate_hits, title=True)
            timings['rerank_ms'] = (perf_counter() - start) * 1000
        else:
            ranked = candidate_hits
        result = {'evidence': ranked[:top_k], 'reranked_candidates': ranked, 'snapshot_id': manifest['snapshot_id'],
                  'route': route, 'bm25_query': lexical_text,
                  'bm25_query_source': 'provided' if lexical_query is not None else 'original_question',
                  'rerank_input': 'title + body' if rerank else None,
                  'timings': {k: round(v, 2) for k, v in timings.items()}, 'candidate_counts': counts}
        if collect_trace:
            result['rankings'] = {**branches, 'union': hits, 'rrf20': candidate_hits, 'after_top5': ranked[:top_k]}
        return result

    def ask(self, question, *, mode='hybrid', top_k=5, rerank=True, generate_answer=True, paper_ids=None,
            lexical_query=None, auto_route=True):
        start = perf_counter()
        result = self.retrieve(question, mode, top_k, rerank, paper_ids,
                               lexical_query=lexical_query, auto_route=auto_route)
        evidence, context_chars = select_context(result['reranked_candidates'], self.settings.context_char_budget)
        result.update(evidence=evidence, answer=None, generation_status='disabled',
                      retrieval_mode=mode, rerank=rerank, context_chars=context_chars,
                      trace_id=uuid4().hex, mode='retrieval_only')
        result['message'] = '已返回检索证据。'
        if generate_answer:
            config = generation_settings(self.root)
            if not evidence:
                result.update(generation_status='no_evidence', message='未找到可用证据，请调整问题或论文范围。')
            elif not config.configured:
                result.update(generation_status='key_required', message='检索已完成；请在 .env 填写 DEEPSEEK_API_KEY 后启用回答生成。')
            else:
                generation_start = perf_counter()
                try:
                    answer = generate(question, evidence, config)
                    result.update(**answer, generation_status='completed', mode='rag', message='回答已生成；引用编号检查通过，内容支持程度仍需核对。')
                except (RuntimeError, ValueError) as error:
                    result.update(generation_status='error', message=str(error))
                result['timings']['generation_ms'] = round((perf_counter() - generation_start) * 1000, 2)
        result['elapsed_ms'] = round((perf_counter() - start) * 1000, 2)
        result['stages'] = [{'name': mode.upper() + ' 检索', 'status': 'done'},
                            {'name': 'BGE 重排', 'status': 'done' if rerank else 'skipped'},
                            {'name': 'DeepSeek 生成', 'status': 'done' if result['generation_status'] == 'completed' else 'pending'}]
        atomic_json(self.root / 'runtime/traces' / (result['trace_id'] + '.json'),
                    {'question': question, 'settings': self.settings.model_dump(), **result})
        return result
