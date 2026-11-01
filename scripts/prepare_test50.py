"""Expand the evaluated 23-question set to 50 source-grounded questions.

Original heldout/dev files stay immutable. New labels use original PDFs only;
no retrieval/model scores select evidence. Old/fresh cohorts are explicit.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_source_pages
from rag.config import load_settings
from rag.ingestion import current_manifest, paper_catalog
from rag.storage import atomic_json, atomic_text
from scripts.prepare_paper_benchmark import locate, folded

LL = '2024.acl-long.91'
HY = '2024.emnlp-industry.66'
PC = '2025.emnlp-industry.20'
OG = '2025.emnlp-main.1674'
PAM = '2024.acl-long.736'
SAFE = '2025.naacl-long.281'
RH = '2025.acl-long.826'
DR = '2025.emnlp-main.1793'

# Answer points and immutable source anchors, authored before any new retrieval.
FACTS = {
    'll_coarse_why': ('整篇文档含较多无关信息，计算给定问题时整篇文档的困惑度可能无法充分区分相关性。', [
        (LL, 3, 'One approach to improve key information density', 'metric for document-level compression.')]),
    'll_coarse_score': ('LongLLMLingua 用给定不同文档时问题的困惑度 p(xque|xdoc) 衡量文档与问题的关联。', [
        (LL, 3, 'We propose to use the perplexity of the question', 'represent the association between them.')]),
    'll_reorder': ('粗粒度压缩后按文档重要性分数重排，利用模型对不同位置的信息感知差异，缓解重要信息处于中部时的退化。', [
        (LL, 4, 'As demonstrated in Figure 1b', 'information perception difference in positions:')]),
    'll_fine': ('直接把问题放在上下文前可能使相关token困惑度降低而难以区分；细粒度压缩用问题条件造成的分布偏移，即contrastive perplexity，衡量token关联性。', [
        (LL, 4, 'A straightforward solution for the awareness', 'the token and the question.')]),
    'hy_route': ('SELF-ROUTE 先给LLM问题和检索块；可回答则接受RAG答案，不可回答则把完整上下文交给长上下文LLM。', [
        (HY, 4, 'Concretely, our method consists of two steps', 'the final prediction (i.e., LC).')]),
    'hy_k': ('增大k增加RAG与路由成本，也可能减少转入长上下文的查询；最优k随数据集、任务性质及性能要求变化。', [
        (HY, 5, 'This is because when k', 'applying our method to various applications.')]),
    'hy_signal': ('SELF-ROUTE 的路由信号来自LLM自我反思，假设模型能较好校准地判断给定上下文是否可回答。', [
        (HY, 4, 'SELF-ROUTE utilizes LLM itself', 'answerable given provided context.')]),
    'pc_segments': ('ProCut支持预定义逻辑块、按句子或段落边界的结构感知切分，以及LLM驱动的模板切分。', [
        (PC, 2, 'We employ three strategies:', 'templates such as those produced by TextGrad.')]),
    'pc_cost': ('传统Leave-One-Out需要M次调用，Shapley需要2的M次方量级；片段和数据集较大时成本高。', [
        (PC, 3, 'The attribution methods discussed above', 'dataset size are high.')]),
    'pc_loop': ('LLM驱动归因生成有限候选mask，在训练集上评估并把反馈交回LLM迭代改进片段排序；限制候选mask可使调用次数少于M。', [
        (PC, 3, 'that leverages the model’s own semantic', 'high attribution fidelity.')]),
    'pc_prune': ('ProCut保留得分最高的floor(rM)个片段，并保持这些片段的原始顺序；r可按成本或任务性能设置。', [
        (PC, 3, 'After scoring the segments', 'or tuned for task performance.')]),
    'og_build': ('OG-RAG用GPT-4o把切分文档中的领域值填入本体，得到本体约束的事实块。', [
        (OG, 4, 'The first part includes mapping', 'complete prompt is provided in Appendix C.2.')]),
    'og_flat': ('将嵌套事实块展平成超边；超节点是拼接嵌套实体及属性后的键值对，超边连接多个超节点。', [
        (OG, 4, 'We define each flattened factual-block', 'the nested entities and their attributes.')]),
    'og_nodes': ('相关超节点可通过查询与实体属性键的相似度，或查询与属性值的相似度寻找，两组节点共同组成目标集合。', [
        (OG, 5, 'In other words, a hypernode is relevant', 'the query Q is high.')]),
    'og_cover': ('OG-RAG贪心选择覆盖最多未覆盖相关节点的超边，直到达到L条超边或覆盖所有相关节点。', [
        (OG, 5, 'In each iteration, we add the hyperedge', 'all the relevant nodes are covered.')]),
    'pam_qr': ('PAM QA第一步复述问题，把问题作为上下文感知的表示。', [
        (PAM, 3, 'The first step is question repetition', 'to prompt the model.')]),
    'pam_ip': ('第二步预测支持文档的索引而不是逐字引文；还增加仅预测正确文档索引的MRC任务。', [
        (PAM, 3, 'Accordingly, the second step is index prediction', 'supplement.')]),
    'pam_as': ('第三步汇总前面信息，生成最终答案；利用问题和证据索引减少无关信息干扰。', [
        (PAM, 3, 'The third step is to generate', 'attention scale decays with increasing distance.')]),
    'pam_train': ('训练分两阶段：先把上下文窗口扩至8K，再用PAM QA数据训练模型应对lost in the middle。', [
        (PAM, 3, 'We equipped our model', 'called "lost in the middle".')]),
    'safe_scope': ('该安全分析比较11个LLM的RAG和非RAG设置，发现RAG可能降低安全性并改变安全轮廓，安全模型与安全文档的组合也可能产生不安全输出。', [
        (SAFE, 1, 'We conduct a detailed comparative analysis', 'can cause unsafe generations.')]),
    'safe_judge': ('文档安全性由Llama Guard 2和Llama-3-70B共同判断，两者都认为不安全才标为不安全；5篇检索文档中至少一篇不安全则文档集合不安全。', [
        (SAFE, 5, 'We also use Llama Guard 2', 'classified as unsafe.')]),
    'rh_neg': ('RHIO通过屏蔽retrieval heads构造模拟真实错误的不忠实回答；普通实体置换可能造成不连贯和较低错误覆盖。', [
        (RH, 3, 'Prior approaches (Mishra et al., 2024)', 'masking out retrieval heads of LLMs.')]),
    'rh_control': ('[POS]引导忠实回答，[NEG]诱导不忠实回答。', [
        (RH, 4, 'During the inference phase', 'when prepending [NEG].')]),
    'rh_sid': ('Self-Induced Decoding放大忠实预测并压低诱导的不忠实预测，以二者的对比改变下一个token概率。', [
        (RH, 4, 'Inspired by studies on contrastive decoding', 'which can be formed as:')]),
    'dr_bridge': ('D-RAG对目标函数重写，并使子图采样及提示构造可微，以联合优化检索器和生成器。', [
        (DR, 4, 'D-RAG constructs differentiable bridges', 'reasoning capabilities.')]),
    'dr_sample': ('D-RAG用Gumbel-Softmax重参数化使离散子图采样可微；事实选择概率由检索器给出。', [
        (DR, 5, 'To make this subgraph sampling process', 'We apply the Gumbel-Softmax trick:')]),
    'dr_semantic': ('语义路径把事实转成head、relation、tail自然语言并编码，用选择指示保留被选事实的嵌入。', [
        (DR, 5, 'For semantic information', 'embeddings of selected facts.')]),
    'dr_structural': ('结构路径将GNN事实表示通过两层MLP投影到LLM空间，选取事实后与语义嵌入拼接成神经事实提示。', [
        (DR, 5, 'For structural information', 'neural fact prompt VF for the LLM-based generator.')]),
}

QUESTIONS = [
    ('LongLLMLingua粗粒度压缩为什么不直接采用整篇文档在给定问题时的困惑度？它采用什么关联度量？',
     'Why does LongLLMLingua avoid document perplexity conditioned on the question for coarse compression, and what association measure does it use?', ['ll_coarse_why', 'll_coarse_score']),
    ('LongLLMLingua粗粒度压缩后如何重排文档？这种排序针对什么位置问题？',
     'How does LongLLMLingua reorder documents after coarse compression, and what position issue does this address?', ['ll_reorder']),
    ('LongLLMLingua细粒度压缩为何使用contrastive perplexity？直接把问题拼在上下文前有什么局限？',
     'Why does LongLLMLingua use contrastive perplexity for fine compression, and what limitation arises from simply prepending the question?', ['ll_fine']),
    ('SELF-ROUTE怎样判断使用RAG还是长上下文LLM？两种情况下分别输入哪些材料？',
     'How does SELF-ROUTE choose between RAG and a long-context LLM, and which inputs are used in each case?', ['hy_route']),
    ('SELF-ROUTE增加检索块数量k如何影响成本？最优k是否对所有任务相同？',
     'How does increasing the retrieved chunk count k affect SELF-ROUTE cost, and is the optimal k the same for every task?', ['hy_k']),
    ('有没有让模型先判断检索块能否回答、不能时再使用完整上下文的路由方法？这种判断依赖什么校准假设？',
     'Is there a routing method where the model first judges whether retrieved chunks can answer and otherwise uses full context? What calibration assumption supports this judgment?', ['hy_route', 'hy_signal']),
    ('ProCut支持哪三种提示模板切分策略？各自依据是什么？',
     'Which three prompt template segmentation strategies does ProCut support, and what guides each strategy?', ['pc_segments']),
    ('ProCut的LLM驱动归因如何利用候选mask和评估反馈？为什么提出它替代昂贵的传统归因计算？',
     'How does ProCut LLM-driven attribution use candidate masks and evaluation feedback, and why replace expensive traditional attribution computations?', ['pc_cost', 'pc_loop']),
    ('ProCut得到片段得分后如何剪枝？保留数量和片段顺序如何确定？',
     'How does ProCut prune after scoring segments, and how are the retained count and segment order determined?', ['pc_prune']),
    ('OG-RAG如何把领域文档转换为本体约束的超图？超节点和超边分别表示什么？',
     'How does OG-RAG convert domain documents into an ontology-grounded hypergraph, and what do hypernodes and hyperedges represent?', ['og_build', 'og_flat']),
    ('OG-RAG查找相关超节点时，查询可以与哪两类内容比较相似度？',
     'Which two kinds of information can OG-RAG compare with a query to identify relevant hypernodes?', ['og_nodes']),
    ('有没有先找到查询相关节点，再选择少量超边覆盖这些节点作为上下文的方法？每轮选择和停止条件是什么？',
     'Is there a method that first finds query-relevant nodes and then covers them with a small set of hyperedges as context? What are its selection and stopping criteria?', ['og_nodes', 'og_cover']),
    ('Never Lost in the Middle: Mastering Long-Context Question Answering with Position-Agnostic Decompositional Training中的PAM QA包含哪三个步骤？为何预测证据索引？',
     'Which three steps constitute PAM QA in Never Lost in the Middle: Mastering Long-Context Question Answering with Position-Agnostic Decompositional Training, and why predict evidence indexes?', ['pam_qr', 'pam_ip', 'pam_as']),
    ('Never Lost in the Middle: Mastering Long-Context Question Answering with Position-Agnostic Decompositional Training如何安排两阶段训练？扩大窗口后训练什么任务？',
     'How does Never Lost in the Middle: Mastering Long-Context Question Answering with Position-Agnostic Decompositional Training organize its two training stages, and what follows window expansion?', ['pam_train']),
    ('RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models比较了多少个LLM？安全模型配合安全文档是否保证安全输出？',
     'How many LLMs are compared in RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models, and do safe models plus safe documents guarantee safe outputs?', ['safe_scope']),
    ('RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models怎样判定单篇检索文档和整个文档集合不安全？',
     'How does RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models classify an individual retrieved document and a retrieved set as unsafe?', ['safe_judge']),
    ('Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization中的RHIO怎样构造不忠实样本？与普通实体置换有何区别？',
     'How does RHIO in Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization construct unfaithful samples, and how does this differ from entity permutations?', ['rh_neg']),
    ('有没有用[POS]和[NEG]分别诱导忠实与不忠实回答，再对比两种预测改善生成的方法？控制token与解码分别起什么作用？',
     'Is there a method using POS and NEG controls to induce faithful and unfaithful answers and contrasting their predictions to improve generation? What do the controls and decoding do?', ['rh_control', 'rh_sid']),
    ('D-RAG通过哪些设计连接检索器和生成器以实现联合优化？离散子图采样如何获得梯度？',
     'Which designs connect the retriever and generator for joint optimization in D-RAG, and how does discrete subgraph sampling support gradients?', ['dr_bridge', 'dr_sample']),
    ('D-RAG的神经事实提示如何同时保留语义和结构信息？两种信息来自哪里，如何组合？',
     'How does D-RAG neural fact prompting preserve both semantic and structural information, where does each come from, and how are they combined?', ['dr_semantic', 'dr_structural']),
]

CROSS = [
    ('LongLLMLingua与ProCut的压缩单位和选择依据有什么区别？分别说明问题关联性与任务表现归因的作用。',
     'How do LongLLMLingua and ProCut differ in compression units and selection signals? Explain query association versus task-performance attribution.',
     ['ll_coarse_score', 'll_fine', 'pc_segments', 'pc_loop', 'pc_prune']),
    ('SELF-ROUTE与LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering怎样应对检索上下文不足？完整上下文和源长段落分别如何使用？',
     'How do SELF-ROUTE and LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering handle insufficient retrieved context, and how do they use full context versus source long paragraphs?',
     ['hy_route', 'long_map', 'long_dual']),
    ('OG-RAG与D-RAG如何组织图中的检索证据并提供给生成器？本体超边覆盖与可微子图及神经提示是否同一种机制？',
     'How do OG-RAG and D-RAG organize graph evidence for generation? Are ontology hyperedge coverage and differentiable subgraphs with neural prompts the same mechanism?',
     ['og_flat', 'og_cover', 'dr_bridge', 'dr_semantic', 'dr_structural']),
    ('RHIO与RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models分别关注什么质量问题？能否把上下文忠实性直接等同于安全性？',
     'Which quality concerns do RHIO and RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models address, and can contextual faithfulness be equated with safety?',
     ['rh_neg', 'rh_sid', 'safe_scope']),
    ('Never Lost in the Middle: Mastering Long-Context Question Answering with Position-Agnostic Decompositional Training与LongLLMLingua如何缓解位置偏差？对比PAM QA训练与文档重排压缩的处理方式。',
     'How do Never Lost in the Middle: Mastering Long-Context Question Answering with Position-Agnostic Decompositional Training and LongLLMLingua address position bias? Compare PAM QA training with document reordering and compression.',
     ['pam_train', 'pam_qr', 'pam_ip', 'll_reorder', 'll_fine']),
]


def annotation():
    return {'method': 'agent_source_grounded', 'human_verified': False,
            'retrieval_used_to_select_gold': False, 'status': 'complete_source_checked',
            'notes': ['新增问题未依据检索分数选择；交叉问题可共享证据，题目并非统计独立样本。']}


def validate_rows(rows, pages, dev_rows):
    ids, texts, refs = set(), set(), 0
    dev_pids = {p for q in dev_rows for p in q['gold_paper_ids']}
    for q in rows:
        if q['question_id'] in ids or folded(q['question'])[0] in texts:
            raise ValueError('Duplicate question or ID')
        ids.add(q['question_id']); texts.add(folded(q['question'])[0])
        if not all(q[k].strip() for k in ['question', 'question_en', 'reference_answer']):
            raise ValueError('Empty question/translation/reference')
        if set(q['gold_paper_ids']) & dev_pids:
            raise ValueError('Test sources overlap development sources')
        units = {u['unit_id']: u for u in q['evidence_units']}
        if len(units) != len(q['evidence_units']) or bool(units) != q['answerable']:
            raise ValueError('Invalid evidence-unit schema')
        for point in q['required_answer_points']:
            if not point['text'] or not set(point['evidence_unit_ids']) <= units.keys():
                raise ValueError('Answer point references missing evidence')
        seen_pids = set()
        for unit in units.values():
            if not unit['alternatives'] or not all(unit['alternatives']):
                raise ValueError('Empty evidence alternative')
            for alternative in unit['alternatives']:
                for span in alternative:
                    key = (span['paper_id'], span['document_sha256'], span['pdf_page'])
                    text = pages[key]
                    a, b = span['char_start'], span['char_end']
                    if not 0 <= a < b <= len(text) or text[a:b] != span['quote']:
                        raise ValueError('Evidence quote/offset differs from source')
                    if hashlib.sha256(span['quote'].encode()).hexdigest() != span['quote_sha256']:
                        raise ValueError('Quote hash differs')
                    seen_pids.add(span['paper_id']); refs += 1
        if q['answerable'] and seen_pids != set(q['gold_paper_ids']):
            raise ValueError('Evidence paper IDs differ')
        if not q['answerable']:
            check = q['absence_check']
            scoped = [t for (p, sha, _), t in pages.items() if p == check['paper_id'] and sha == check['document_sha256']]
            if len(scoped) != check['pages_checked'] or folded(check['absent_term'])[0] in folded('\n'.join(scoped))[0]:
                raise ValueError('Unsupported-term absence check failed')
    return refs


def main():
    target = ROOT/'data/evaluation/paper_benchmark'
    old = [json.loads(s) for s in (target/'heldout.jsonl').read_text(encoding='utf-8').splitlines() if s]
    dev = [json.loads(s) for s in (target/'dev.jsonl').read_text(encoding='utf-8').splitlines() if s]
    if len(old) != 23 or len(QUESTIONS) != 20 or len(CROSS) != 5:
        raise ValueError('Expected original 23 plus 20 single, 5 cross and 2 negative questions')
    catalog = {p['paper_id']: p for p in paper_catalog()}
    ids = {p for q in old for p in q['gold_paper_ids']} | {s[0] for _, specs in FACTS.values() for s in specs}
    pages = load_source_pages(ids)
    manifest = current_manifest(load_settings())
    if any(manifest['documents'].get(pid) != sha for pid, sha, _ in pages):
        raise ValueError('PDF does not match frozen indexed corpus')
    units = {u['unit_id']: u for q in old for u in q['evidence_units']}
    for name, (point, specs) in FACTS.items():
        spans = []
        for pid, page, begin, finish in specs:
            key = next(k for k in pages if k[0] == pid and k[2] == page)
            text = pages[key]
            try:
                a, _ = locate(text, begin)
                _, b = locate(text, finish, after=a)
            except ValueError as e:
                raise ValueError(f'{name}: {e}') from None
            quote = text[a:b]
            spans.append({'paper_id': pid, 'document_sha256': key[1], 'title': catalog[pid]['title'],
                          'source_url': catalog[pid]['landing_url'], 'pdf_page': page,
                          'char_start': a, 'char_end': b, 'quote': quote,
                          'quote_sha256': hashlib.sha256(quote.encode()).hexdigest()})
        units[name] = {'unit_id': name, 'answer_point': point, 'alternatives': [spans]}
    added = []
    for category, specs in [('single_paper', QUESTIONS), ('cross_paper', CROSS)]:
        for number, (zh, en, names) in enumerate(specs, 1):
            evidence = [units[name] for name in names]
            scope = 'open_corpus_anchor' if category == 'single_paper' and number in {6, 12, 18} else 'named_papers'
            added.append({'question_id': f'test50_{"single" if category=="single_paper" else "cross"}{number:02d}',
                          'split': 'test50', 'category': category, 'question': zh, 'question_en': en,
                          'language': 'zh', 'answerable': True, 'source_scope': scope,
                          'gold_paper_ids': sorted({s['paper_id'] for u in evidence for a in u['alternatives'] for s in a}),
                          'gold_sources_exhaustive': scope == 'named_papers', 'evidence_spans_exhaustive': False,
                          'reference_answer': ''.join(u['answer_point'] for u in evidence),
                          'required_answer_points': [{'point_id': u['unit_id'], 'text': u['answer_point'],
                                                      'evidence_unit_ids': [u['unit_id']]} for u in evidence],
                          'evidence_units': evidence, 'annotation': annotation()})
    negatives = [
        (LL, 'LongLLMLingua原论文在DeepSeek V4 Pro上压缩4倍后报告的F1是多少？只依据原论文给出数值。',
         'What F1 does the original LongLLMLingua paper report for four-fold compression with DeepSeek V4 Pro? Answer only from the paper.', 'DeepSeek V4 Pro'),
        (PC, 'ProCut原论文报告在RTX 5090上处理工业提示模板的压缩耗时是多少毫秒？只依据该论文回答。',
         'What compression latency in milliseconds does the original ProCut paper report for industrial templates on an RTX 5090? Answer only from that paper.', '5090')]
    for number, (pid, zh, en, absent) in enumerate(negatives, 1):
        document = [t for (p, _, _), t in pages.items() if p == pid]
        if folded(absent)[0] in folded('\n'.join(document))[0]:
            raise ValueError('Requested absent term is actually present')
        sha = next(k[1] for k in pages if k[0] == pid)
        added.append({'question_id': f'test50_negative{number:02d}', 'split': 'test50', 'category': 'unsupported',
                      'question': zh, 'question_en': en, 'language': 'zh', 'answerable': False,
                      'source_scope': 'named_papers', 'gold_paper_ids': [pid], 'gold_sources_exhaustive': True,
                      'evidence_spans_exhaustive': False, 'reference_answer': '当前指定论文未提供所问条件下的结果，不能编造数值。',
                      'required_answer_points': [{'point_id': 'insufficient_information',
                                                 'text': '说明证据不足，不编造数值或替换实验条件。', 'evidence_unit_ids': []}],
                      'evidence_units': [], 'absence_check': {'paper_id': pid, 'document_sha256': sha,
                          'pages_checked': len(document), 'absent_term': absent, 'method': 'full_text_scope_check',
                          'scope_note': '只检查指定论文的文本，不代表整个论文库不存在该结果。'}, 'annotation': annotation()})
    rows = old + added
    refs = validate_rows(rows, pages, dev)
    if len(rows) != 50:
        raise ValueError('Expected exactly 50 detection questions')
    data = ''.join(json.dumps(q, ensure_ascii=False)+'\n' for q in rows)
    sha = hashlib.sha256(data.encode()).hexdigest()
    atomic_text(target/'test50.jsonl', data)
    atomic_json(target/'test50_manifest.json', {'version': 'test50-v1', 'dataset_sha256': sha,
                'questions': 50, 'answerable': sum(q['answerable'] for q in rows),
                'categories': dict(Counter(q['category'] for q in rows)), 'source_papers': len(ids),
                'scopes': dict(Counter(q['source_scope'] for q in rows)),
                'new_source_papers': sorted(ids - {p for q in old for p in q['gold_paper_ids']}),
                'cohorts': {'previously_evaluated23': [q['question_id'] for q in old],
                            'new_questions27': [q['question_id'] for q in added]},
                'original_heldout_sha256': hashlib.sha256((target/'heldout.jsonl').read_bytes()).hexdigest(),
                'evidence_span_references_checked': refs, 'all_quotes_offsets_hashes_valid': True,
                'development_source_overlap': 0, 'annotation_method': 'agent_source_grounded', 'human_verified': False,
                'limitations': 'Includes 23 previously evaluated questions. Report new 27 separately. Related questions can share evidence, so 50 questions are not 50 statistically independent samples. No new retrieval scores used to author labels.'})
    print('Prepared 50 questions:', sum(q['answerable'] for q in rows), 'answerable,', sum(not q['answerable'] for q in rows), 'unsupported')
    print('Original 23 preserved; new 27; new papers 8; source papers', len(ids))
    print('Validated evidence references:', refs)
    print('Dataset:', target/'test50.jsonl')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
