"""Question-only semantic decomposition with validated, credential-free caching."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from rag.config import ROOT
from rag.storage import atomic_json


PROMPT_VERSION = 'question-facets-zh-en-v1'
MAX_SUBQUERIES = 3
SYSTEM_PROMPT = '''你负责为论文检索拆解问题，不回答问题。
用户提供中文原问题和固定英文译文。将复杂问题拆成最多3个独立、具体、互补的检索子问题。
子问题只重新组织用户已经提出的信息需求。保留必要的论文名、方法名和指代对象。
跨论文问题可以分别问每篇论文，但不得把两篇论文的条件混在一起。
开放式问题保持开放，禁止从记忆补入用户未指定的论文或方法名。
禁止补写答案、实验数字、假设的技术步骤或未出现的新事实。
不要仅对原问题做同义改写；简单问题不必拆解，subqueries返回空数组。
每个子问题同时提供中文zh和英文en。英文用于检索英文论文，中文用于向量检索。
reason用一句话说明拆分依据，不输出推理过程。严格输出JSON：
{"subqueries":[{"zh":"中文子问题","en":"English subquestion"}],"reason":"简短说明"}'''


class Subquery(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    zh: str = Field(min_length=1, max_length=500)
    en: str = Field(min_length=1, max_length=800)


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    subqueries: list[Subquery] = Field(max_length=MAX_SUBQUERIES)
    reason: str = Field(min_length=1, max_length=400)


def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()


def validate_plan(plan, question, question_en, router):
    seen_zh, seen_en = {normalized(question)}, {normalized(question_en)}
    original_ids = set(router.route(question)['matched_paper_ids']) | set(router.route(question_en)['matched_paper_ids'])
    queries = []
    for query in plan.subqueries:
        zh, en = query.zh.strip(), query.en.strip()
        if not zh or not en:
            raise ValueError('拆解结果含空白子问题。')
        # Duplicates of the original or another facet consume no retrieval slots.
        if normalized(zh) in seen_zh or normalized(en) in seen_en:
            continue
        mentioned = set(router.route(zh)['matched_paper_ids']) | set(router.route(en)['matched_paper_ids'])
        if mentioned - original_ids:
            raise ValueError('拆解引入了原问题没有指定的已知论文，请检查拆解结果。')
        queries.append(Subquery(zh=zh, en=en))
        seen_zh.add(normalized(zh))
        seen_en.add(normalized(en))
    return QueryPlan(subqueries=queries, reason=plan.reason.strip())


def cache_identity(question, question_en, settings, router):
    identity = {'question': question, 'question_en': question_en, 'prompt_version': PROMPT_VERSION,
                'model': settings.model, 'base_url': settings.base_url.rstrip('/'),
                'catalog_fingerprint': router.fingerprint}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_or_create_plan(question, question_en, settings, router, *, create=False, root=ROOT, transport=None):
    """One API request per uncached plan only when create=True; no implicit retries."""
    identity = cache_identity(question, question_en, settings, router)
    path = Path(root)/'runtime/query_plans'/(identity+'.json')
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['cache_identity'] != identity or record['question'] != question or record['question_en'] != question_en:
            raise ValueError('拆解缓存与当前问题不匹配。')
        plan = validate_plan(QueryPlan.model_validate(record['plan']), question, question_en, router)
        return plan, {**record, 'cache_hit': True, 'cache_path': str(path)}
    if not create:
        return None, {'cache_hit': False, 'cache_path': str(path), 'cache_identity': identity}
    if not settings.configured:
        raise RuntimeError('请在本地 .env 填写 DEEPSEEK_API_KEY，然后使用 --prepare 生成拆解缓存。')
    payload = {'model': settings.model, 'temperature': 0, 'max_tokens': 900,
               'thinking': {'type': 'disabled'}, 'response_format': {'type': 'json_object'},
               'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                            {'role': 'user', 'content': json.dumps({'question': question, 'question_en': question_en}, ensure_ascii=False)}]}
    with httpx.Client(timeout=httpx.Timeout(45, connect=10), follow_redirects=False, transport=transport) as client:
        try:
            response = client.post(settings.base_url.rstrip('/')+'/chat/completions', json=payload,
                                   headers={'Authorization': 'Bearer '+settings.api_key.get_secret_value()})
        except httpx.HTTPError:
            raise RuntimeError('DeepSeek 拆解请求失败，请检查网络后重试；本次没有自动重试。') from None
    if response.is_error:
        raise RuntimeError(f'DeepSeek 拆解请求返回 HTTP {response.status_code}，请检查本地模型配置、密钥或余额。')
    try:
        body = response.json()
        choice = body['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('拆解输出未完整结束。')
        plan = validate_plan(QueryPlan.model_validate_json(choice['message']['content']), question, question_en, router)
    except (KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError('拆解结果未通过格式或论文实体校验；没有写入缓存，请重试或检查模型配置。') from None
    record = {'cache_identity': identity, 'question': question, 'question_en': question_en,
              'prompt_version': PROMPT_VERSION, 'request_model': settings.model,
              'response_model': body.get('model', settings.model), 'usage': body.get('usage', {}),
              'plan': plan.model_dump()}
    atomic_json(path, record)
    return plan, {**record, 'cache_hit': False, 'cache_path': str(path)}


def allocate_slots(total, query_count):
    """Reserve half for the original; evenly divide the rest among facets."""
    if total < query_count or query_count < 1:
        raise ValueError('Not enough retrieval slots for these queries')
    if query_count == 1:
        return [total]
    original = (total+1)//2
    count = query_count-1
    base, extra = divmod(total-original, count)
    if base == 0:
        raise ValueError('No retrieval slot left for a facet')
    return [original, *[base+(i < extra) for i in range(count)]]
