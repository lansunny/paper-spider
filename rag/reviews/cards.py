"""Four sourced topic cards, organized by existing method-community intersections."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json

from rag.formula_store import digest_file
from rag.generation import generation_settings
from rag.ingestion import fingerprint
from rag.paper_graph.extraction import request_json
from rag.storage import atomic_json, read_json
from .corpus import ReviewCorpus
from .schema import CommunityCard

CARD_VERSION = 'topic-method-research-cards-v1'
CARD_SYSTEM = '''你为个人论文库整理带来源的研究卡片。仅使用输入中的 claims，不调用记忆补充事实。
输入是系统分组线索，不是公认学术分类。社区相似不代表引用、继承、因果或实验结果可比。
输出 JSON：{"title":"候选主题名称","scope":[statement],"routes":[route]}。
statement={"text":"具体且有区分度的陈述","support":"multi_paper 或 paper_specific","claim_ids":["C1"]}。
multi_paper 必须引用至少两篇不同论文的 claims；paper_specific 只引用同一篇论文。
不能把一篇论文的发现写成整个群组的共识；只说明实际支持该陈述的论文。
route={"method_community_id":"输入的分组ID","name":"候选技术路线名","overview":statement,
"differences":[statement],"experiments":[statement],"limitations":[statement],"representative_paper_ids":["输入论文ID"]}。
每个输入 route_cell 都输出一个 route，不得新增或漏掉分组。每路线概述约80字，差异、实验、局限各0到1条。
各 route 的 claim_ids 必须属于该 route_cell 的成员。代表论文只能从该 cell 的成员选择，最多2篇。
材料不足的 differences/experiments/limitations 留空，不编造信息、数值、优劣或对比条件。
scope 输出1到2条，保持克制。路线名称用于导航，不宣称是领域完整分类。只返回严格 JSON。'''


def card_input(corpus, topic):
    claims, aliases = [], {}
    limits = {'topic': 1, 'method': 3, 'experiment': 2, 'conclusion': 2}
    for pid in topic['members']:
        for dimension, limit in limits.items():
            values = corpus.paper_claims(pid, dimension)
            if dimension == 'conclusion':
                values = sorted(values, key=lambda c: c['field'] != 'limitations')
            for claim in values[:limit]:
                alias = 'C'+str(len(aliases)+1)
                aliases[alias] = claim['claim_id']
                claims.append({'id': alias, 'paper_id': pid, 'dimension': dimension,
                               'field': claim['field'], 'text': claim['text']})
    return {'topic_community_id': topic['community_id'],
        'papers': [{'paper_id': p, 'title': corpus.papers[p]['title']} for p in topic['members']],
        'route_cells': corpus.route_cells(topic['community_id']), 'claims': claims}, aliases


def validate_card(value, corpus, payload, aliases):
    card = CommunityCard.model_validate(value)
    cells = {c['method_community_id']: c for c in payload['route_cells']}
    if len(card.routes) != len(cells) or {r.method_community_id for r in card.routes} != set(cells):
        raise ValueError('Return exactly one route per provided method community')

    def checked(statement, members):
        if not set(statement.claim_ids) <= set(aliases) or len(set(statement.claim_ids)) != len(statement.claim_ids):
            raise ValueError('Use unique claim IDs copied from the input')
        canonical = [aliases[c] for c in statement.claim_ids]
        papers = {corpus.claims[c]['paper_id'] for c in canonical}
        if not papers <= set(members):
            raise ValueError('A route statement cites a claim from another route')
        if statement.support == 'multi_paper' and len(papers) < 2:
            raise ValueError('multi_paper requires claims from at least two distinct papers')
        if statement.support == 'paper_specific' and len(papers) != 1:
            raise ValueError('paper_specific must cite exactly one paper')
        if any(term in statement.text for term in ('所有论文','全部成员','全体论文','所有研究')) and papers != set(members):
            raise ValueError('Do not generalize a subset to all community members')
        return {**statement.model_dump(), 'claim_ids': canonical, 'supporting_paper_ids': sorted(papers),
                'source_ids': sorted({s for c in canonical for s in corpus.claims[c]['source_ids']})}

    output = {'title': card.title, 'topic_community_id': payload['topic_community_id'],
              'paper_ids': [p['paper_id'] for p in payload['papers']], 'scope': [], 'routes': []}
    output['scope'] = [checked(s, output['paper_ids']) for s in card.scope]
    for route in card.routes:
        cell = cells[route.method_community_id]
        if not set(route.representative_paper_ids) <= set(cell['paper_ids']):
            raise ValueError('Representative papers must belong to the route')
        output['routes'].append({**cell, 'name': route.name,
            'representative_paper_ids': route.representative_paper_ids,
            'overview': checked(route.overview, cell['paper_ids']),
            **{field: [checked(s, cell['paper_ids']) for s in getattr(route, field)]
               for field in ('differences','experiments','limitations')}})
    return output


def request_step(system, payload, settings, root, validator, purpose):
    identity = fingerprint({'system': system, 'payload': payload, 'model': settings.model,
                            'endpoint': settings.base_url, 'max_tokens': settings.max_tokens})
    directory = root/'runtime/reviews/cache/model_steps'
    cached = (directory/(identity+'.json')).exists()
    value, record = request_json(system, payload, settings, directory, validator, local_aliases=False,
        repair=lambda error: '格式或来源校验失败：'+str(error)[:1400]+
        '。只按 system 规定的原 JSON 结构重答，不新增字段。编号必须从输入复制；'
        'multi_paper 需至少两篇论文，paper_specific 只能一篇；证据不足的内容留空，不编造来源。')
    usage = {key: sum(a.get('usage', {}).get(key, 0) for a in record['attempts'])
             for key in ('prompt_tokens','completion_tokens','total_tokens')}
    return value, {'purpose': purpose, 'request_sha256': identity, 'cache_hit': cached, 'usage': usage,
                   'model': record['attempts'][-1].get('model', settings.model) if record['attempts'] else settings.model}


def build_cards(root, progress=None):
    progress = progress or (lambda **state: None)
    corpus = ReviewCorpus(root)
    for paper_id in corpus.papers:
        corpus.canonical_pages(paper_id)
    settings = generation_settings(root).model_copy(update={'max_tokens':6500, 'timeout_seconds':180})
    if not settings.configured:
        raise RuntimeError('请先在本地 .env 中设置 DEEPSEEK_API_KEY。')
    cardset_id = fingerprint([CARD_VERSION, corpus.build_id, settings.model, settings.base_url, CARD_SYSTEM])[:24]
    target = root/'runtime/reviews/cards'/cardset_id
    cards, steps = [], []
    def make(topic):
        payload, aliases = card_input(corpus, topic)
        card, step = request_step(CARD_SYSTEM, payload, settings, root,
            lambda value: validate_card(value, corpus, payload, aliases), 'community_card')
        return card, step
    progress(stage='cards', completed=0, total=len(corpus.communities['topic']))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(make, topic) for topic in corpus.communities['topic']]
        for future in as_completed(futures):
            card, step = future.result()
            cards.append(card)
            steps.append(step)
            progress(stage='cards', completed=len(cards), total=len(futures))
    corpus.assert_current()
    cards.sort(key=lambda c: c['topic_community_id'])
    atomic_json(target/'cards.json', cards)
    manifest = {'status':'completed', 'cardset_id':cardset_id, 'graph_build_id':corpus.build_id,
        'version':CARD_VERSION, 'created_at':datetime.now(timezone.utc).isoformat(),
        'papers':len(corpus.papers), 'topic_cards':len(cards), 'method_routes':sum(len(c['routes']) for c in cards),
        'cards_sha256':digest_file(target/'cards.json'), 'steps':steps,
        'validation':'claim_ids_membership_support_scope_only_not_semantic_entailment',
        'classification_status':'source_grounded_candidate_routes', 'model':settings.model}
    atomic_json(target/'manifest.json', manifest)
    atomic_json(root/'runtime/reviews/CURRENT.json', {'directory':target.relative_to(root).as_posix(),
                'manifest_sha256':digest_file(target/'manifest.json')})
    return manifest


def load_cards(root, corpus):
    pointer = read_json(root/'runtime/reviews/CURRENT.json')
    if not pointer:
        raise FileNotFoundError('请先准备研究卡片。')
    directory = (root/pointer['directory']).resolve()
    if not directory.is_relative_to((root/'runtime/reviews/cards').resolve()):
        raise ValueError('Invalid research card directory')
    if digest_file(directory/'manifest.json') != pointer['manifest_sha256']:
        raise ValueError('Research card manifest changed')
    manifest = read_json(directory/'manifest.json')
    if manifest['graph_build_id'] != corpus.build_id:
        raise ValueError('论文图已更新，请重新整理研究卡片。')
    if digest_file(directory/'cards.json') != manifest['cards_sha256']:
        raise ValueError('Research cards changed')
    return manifest, json.loads((directory/'cards.json').read_text(encoding='utf-8'))
