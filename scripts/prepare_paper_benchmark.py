"""Build source-grounded labels, independently of retrieval/model scores."""
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.storage import atomic_json, atomic_text

RAGAS = '2024.eacl-demo.16'
ARES = '2024.naacl-long.20'
FLARE = '2023.emnlp-main.495'
IRCOT = '2023.acl-long.557'
DPR = '2020.emnlp-main.550'
ROCKET = '2021.emnlp-main.224'
REWRITE = '2023.emnlp-main.322'
RAFE = '2024.findings-emnlp.49'
QASPER = '2021.naacl-main.365'
SCIDQA = '2024.emnlp-main.1163'
LIGHT = '2025.findings-emnlp.568'
GRAG = '2025.findings-naacl.232'
LONG = '2024.emnlp-main.1259'
SCITAT = '2025.findings-acl.199'
DEV = {RAGAS, ARES, FLARE, IRCOT, DPR, ROCKET}


def folded(text):
    """Anchor lookup only: preserve original PDF text/offsets in resulting labels."""
    chars, offsets = [], []
    for index, char in enumerate(text):
        for value in unicodedata.normalize('NFKC', char).lower():
            if not value.isspace() and value != '-':
                chars.append(value)
                offsets.append(index)
    return ''.join(chars), offsets


def locate(text, anchor, after=0):
    haystack, offsets = folded(text)
    needle, _ = folded(anchor)
    matches, cursor = [], 0
    while True:
        found = haystack.find(needle, cursor)
        if found < 0:
            break
        if offsets[found] >= after:
            matches.append((offsets[found], offsets[found + len(needle) - 1] + 1))
        cursor = found + len(needle)
    if len(matches) != 1:
        raise ValueError(f'Anchor must be unique: {anchor!r}; matches={len(matches)}')
    return matches[0]


def S(paper, page, start, end=None):
    return paper, page, start, end


# Each fact is a required answer point. Its spans jointly support that point.
# Anchors are checked against canonical page text, never inferred from a retriever.
FACTS = {
    'r_statements': ('先用 LLM 将生成回答拆成较短、集中的陈述集合 S。', [
        S(RAGAS, 3, 'To estimate faithfulness, we first', 'shorter and more focused assertions.')]),
    'r_verify': ('再用 LLM 逐条判断陈述能否由检索上下文推出。', [
        S(RAGAS, 3, 'S(as(q)), the LLM determines', 'verification function v(si, c(q)).')]),
    'r_fraction': ('Faithfulness = 被上下文支持的陈述数 / 回答陈述总数，即 |V|/|S|。', [
        S(RAGAS, 3, 'The final faithfulness score', 'the total number of statements in S(as(q)).')]),
    'r_relevance': ('Answer Relevance 关注是否回答问题，不评估事实正确性，并惩罚不完整或冗余回答。', [
        S(RAGAS, 3, 'Answer relevance We say', 'redundant information.')]),
    'r_questions': ('从回答生成 n 个潜在问题；对这些问题与原问题编码，计算余弦相似度并取均值。', [
        S(RAGAS, 3, 'To estimate answer relevance', 'based on as(q), as follows:'),
        S(RAGAS, 3, 'We then obtain embeddings', 'This metric evaluates how closely the generated answer aligns with the initial question or instruction.')]),
    'r_dimensions': ('RAGAs 原论文评估 Faithfulness、Answer Relevance、Context Relevance 三个方面。', [
        S(RAGAS, 3, 'We focus in particular three quality aspects', 'as little irrelevant information as possible.')]),
    'r_reference_free': ('RAGAs 的这些指标可不依赖参考答案；这不等于整个研究没有人工标注。', [
        S(RAGAS, 3, 'datasets or reference answers.', 'fully self-contained and reference-free.')]),
    'r_humans': ('论文用 WikiEval 的人工判断验证指标：两名标注者标注三个维度并讨论分歧。', [
        S(RAGAS, 4, 'All questions were annotated', 'discussion between the annotators.')]),
    'a_inputs': ('ARES 需要域内段落、约 150 条或更多人工偏好验证样本，以及至少 5 个域内问答示例。', [
        S(ARES, 3, 'There are three required inputs:', 'synthetic data generation.')]),
    'a_pipeline': ('ARES 生成合成问答，训练评判器，再用人工验证集配合 PPI 估计各项质量的置信区间。', [
        S(ARES, 3, 'With our inputs prepared', 'quality of each RAG system.')]),
    'a_dimensions': ('ARES 评估 Context Relevance、Answer Faithfulness、Answer Relevance；各指标分别训练带二分类头的评判器。', [
        S(ARES, 3, '1. Context Relevance:', 'classify positive and negative examples.')]),
    'a_ppi': ('PPI 利用人工验证集估计评判器误差，并结合大量未标注数据上的预测构造质量置信区间。', [
        S(ARES, 4, 'To do this, PPI uses', 'simply using annotated outputs from the evaluated RAG system.')]),
    'f_trigger': ('FLAREdirect 先生成临时下一句；任一 token 概率低于阈值 θ 时触发检索，检索后重新生成该句，否则接受临时句。', [
        S(FLARE, 4, 'As shown in Figure 1, at step t', 'then regenerate the next sentence st.'),
        S(FLARE, 4, 'Since LMs tend to be well-calibrated', 'a threshold θ ∈[0, 1].')]),
    'f_query': ('为避免错误临时句污染检索，可屏蔽低置信度 token，或针对低置信度片段生成显式问题作为查询。', [
        S(FLARE, 5, 'Masked sentences as implicit queries.', 'improve retrieval accuracy.'),
        S(FLARE, 5, 'Generated questions as explicit queries.', 'retrieve relevant information.')]),
    'f_instruct': ('FLAREinstruct 则在模型生成 [Search(query)] 指令时暂停生成并检索，将文档前置后继续，直到下一次查询或结束。', [
        S(FLARE, 4, 'As shown in Figure 2, when the LM generates', 'or reaches the end.')]),
    'i_loop': ('IRCoT 先用原问题检索，交替扩展 CoT 和用最后一句 CoT 再检索；已有证据帮助推理，推理帮助寻找下一跳证据。', [
        S(IRCOT, 2, 'We begin by retrieving a base set', 'add to the collected set.')]),
    'i_stop': ('当 CoT 报告答案或达到最大推理步数时停止；将累计段落作为检索结果交给后续 QA。', [
        S(IRCOT, 2, 'We repeat these steps till', 'CoT prompting (Wei et al., 2022).')]),
    'd_negatives': ('DPR 比较随机负例、BM25 高排名但不含答案的负例、其他问题的正段落作为 Gold 负例；最佳方案结合 batch 内 Gold 负例和一个 BM25 负例。', [
        S(DPR, 3, 'We consider three different types of negatives:', 'one BM25 negative passage.')]),
    'd_batch': ('一个 batch 内 Q、P 均为 B×d，QP^T 产生 B×B 分数；对角是正例，其他配对为负例，复用计算。', [
        S(DPR, 3, 'Let Q and P be', 'negative otherwise.')]),
    'd_arch': ('DPR 分别编码问题和段落，用向量点积度量相关性；段落向量可离线预计算并建立检索索引。', [
        S(DPR, 3, 'At run-time, DPR applies', 'dot product of their vectors:'),
        S(DPR, 3, 'Inference During inference time', 'index them using FAISS (Johnson et al., 2017) offline.')]),
    'd_focus': ('DPR 的核心训练目标是改进 dense retriever，并将检索到的段落交给 reader；不同于 RocketQAv2 的 retriever-reranker 联合蒸馏。', [
        S(DPR, 2, 'We focus our research in this work', 'for the reader at run-time.')]),
    'rq_joint': ('RocketQAv2 联合训练 dense retriever 与 reranker，用动态 listwise 蒸馏让两者相互学习，蒸馏过程中两者都更新。', [
        S(ROCKET, 2, 'To this end, we present', 'mutual improvement in joint training.')]),
    'rq_ce': ('Cross-encoder 联合处理 query 与 passage，用 [SEP] 拼接和 [CLS] 线性层打分；能捕获更细语义交互，但比 dual-encoder 耗费更多计算。', [
        S(ROCKET, 3, 'Among the implementations of the re-ranker', 'fed into a learned linear function.')]),
    'w_reward': ('Query Rewriting 将小型可训练改写器放在检索之前，冻结检索器和阅读器，以阅读器下游表现为强化学习奖励。', [
        S(REWRITE, 2, 'We also propose a trainable scheme', 'improve the reader on downstream tasks.')]),
    'w_web': ('该框架使用互联网搜索引擎检索，可免于维护搜索索引并获得较新知识；这描述的是该论文的设计。', [
        S(REWRITE, 2, 'We adopt the off-the-shelf tool', 'up-to-date knowledge (Lazaridou et al., 2022).')]),
    'ra_reward': ('RaFe 使用 reranker 的排序分数作为查询改写训练的反馈，并支持离线和在线 RL。', [
        S(RAFE, 2, 'Subsequently, we utilize', 'both offline and online RL feedback training.')]),
    'ra_cold': ('RaFe 先做监督微调冷启动，改写训练数据由 LLM 生成；免额外反馈标签不等于不用训练数据。', [
        S(RAFE, 2, 'Before leveraging the ranking feedback', 'denoted as Tall')]),
    'q_collection': ('QASPER 共 5,049 问、1,585 篇 NLP 论文；提问者只读标题和摘要，另一组 NLP 从业者读全文作答并给出证据。', [
        S(QASPER, 1, 'We therefore present QASPER', 'provide supporting evidence to answers.')]),
    'q_evidence': ('QASPER 证据可为文本、表格、图片；55.5% 的问题需要多段证据，13% 需要表格或图片。', [
        S(QASPER, 2, 'In addition to providing answers', '13% require tables or figures.')]),
    'q_text_scope': ('QASPER 论文的文本实验忽略需要图表作为证据的 13% 问题，并将多模态 QA 留给后续工作；不能代表完整图表能力。', [
        S(QASPER, 6, 'We restrict our experiments', 'multimodal question answering to future work.')]),
    'sci_source': ('SciDQA 共 2,937 问答对，问题来自领域专家的同行评审，答案来自论文作者。', [
        S(SCIDQA, 1, 'consisting of 2,937 QA pairs.', 'thorough examination of the literature.')]),
    'sci_subset': ('SciDQA 经人工专家编辑与标注，但因为标注成本高，只为数据集的一个子集提供证据。', [
        S(SCIDQA, 2, 'LLM-based QA extraction from peer reviews', 'high annotation cost.')]),
    'sci_multidoc': ('约 11% 的 SciDQA 问题需要除主论文之外至少一篇显式引用论文的推理，还可能涉及图表、公式和参考资料。', [
        S(SCIDQA, 2, 'It is diverse and also some questions', 'candidate paper.')]),
    'l_levels': ('LightRAG 低层检索特定实体及属性、关系；高层聚合多个相关实体与关系，面向主题与全局概念。', [
        S(LIGHT, 4, '• Low-Level Retrieval.', 'rather than specific details.')]),
    'l_matching': ('LightRAG 提取局部与全局关键词，向量匹配实体与关系，并补充检索元素的一跳邻居。', [
        S(LIGHT, 4, '• (i) Query Keyword Extraction.', 'one-hop neighboring nodes of the retrieved nodes v and edges e, respectively.')]),
    'l_answer': ('LightRAG 将相关实体与关系的名称、描述、原文片段等文本整合进 LLM 上下文生成答案。', [
        S(LIGHT, 5, 'Utilization of Retrieved Information.', 'Appendix 9.3).')]),
    'l_update': ('新文档按相同图索引流程处理，再与已有节点、边合并；论文设计无需完整重建整个索引图。', [
        S(LIGHT, 4, 'before, resulting in', 'the edge sets ˆE and ˆE′.'),
        S(LIGHT, 4, 'Reducing Computational Overhead', 'assimilation of new data.')]),
    'l_setup': ('LightRAG 在 UltraDomain 的 Agriculture、CS、Legal、Mixed 上实验，以 LLM 评判全面性、多样性、赋能性及 Overall。', [
        S(LIGHT, 5, 'To evaluate the effectiveness of LightRAG', 'Comprehensiveness, Diversity, Empowerment, and Overall.')]),
    'l_model': ('LightRAG 默认所有 LLM 操作使用 GPT-4o-mini；比较两份答案并计算胜率，非同一基准上的答案 F1。', [
        S(LIGHT, 11, 'For all LLM-based operations', 'GPT-4o-mini.'),
        S(LIGHT, 11, 'The LLM directly compares two answers', 'leading to the final results.')]),
    'g_retrieve': ('GRAG 先检索相关 ego-graphs，再以 graph soft pruning 精炼并合并，近似寻找相关文本子图，避免穷举。', [
        S(GRAG, 2, 'Specifically, to achieve efficient textual subgraph retrieval', 'Johnson and Garey, 1979).')]),
    'g_prompt': ('GRAG 同时输入硬提示（层次化文本图描述）与软提示（图编码器产生的图 token），保留文本和拓扑信息。', [
        S(GRAG, 2, 'To integrate the retrieved textual subgraphs', 'text information as node/edge attributes in graphs.')]),
    'long_dual': ('LongRAG 用 Long-Context Extractor 从源段落提取全局信息，CoT-guided Filter 从检索块筛选事实细节，二者共同供生成器使用。', [
        S(LONG, 3, 'Following the structure of Vanilla RAG', 'generate a final answer to the question.')]),
    'long_map': ('LongRAG 将检索到的短块映射回源长段落，以恢复背景、结构与上下文，随后交给信息提取器。', [
        S(LONG, 5, 'To address these issues, we map', 'using a mapping function fm(·):'),
        S(LONG, 5, 'This mapping strategy maximizes', 'employing zero-shot ICL.')]),
    'long_setup': ('这篇 LongRAG 在 LongBench 的 HotpotQA、2WikiMultiHopQA、MusiQue 上报告 F1；模型包括带 SFT 的开源模型及未 SFT 的 GPT-3.5、GLM-4 等。', [
        S(LONG, 6, 'Datasets & Evaluation.', 'metrics for all three datasets.'),
        S(LONG, 6, '# Ours with SFT #', 'Table 1: Results (%)')]),
    't_need': ('SCITAT 强调表格与正文的联系：只分开处理会漏掉关联信息，问题尽可能同时涉及二者。', [
        S(SCITAT, 1, 'Secondly, prior works focus', 'considering tables and text simultaneously.'),
        S(SCITAT, 1, 'To ensure the questions require reasoning', 'tables and text as much as possible.')]),
    't_types': ('SCITAT 共 953 问、871 篇论文，覆盖 4 类、13 子类推理；4 类为 Look Up、Numerical Reasoning、Data Analysis、Tabulation。', [
        S(SCITAT, 1, 'Overall, SCITAT contains', '4 reasoning types and 13 subtypes'),
        S(SCITAT, 1, 'L, N, D, and T denote', 'Tabulation, with examples in Figure 1.')]),
    't_car': ('CAR 包含 Calculator 和 Reasoner：前者从表格与正文提取并计算数字信息，再提供给后者推理。', [
        S(SCITAT, 2, 'To handle multiple reasoning types', 'Reasoner for further reasoning.')]),
}


def main():
    sources = json.loads((ROOT / 'runtime/annotation_sources.json').read_text(encoding='utf-8'))
    catalog = {p['paper_id']: p for p in json.loads((ROOT / 'data/paper_manifest.json').read_text(encoding='utf-8'))}
    facts = {}
    for key, (point, specs) in FACTS.items():
        spans = []
        for pid, page, begin, finish in specs:
            source = sources[pid]
            text = next(p['text'] for p in source['pages'] if p['page'] == page)
            left, begin_end = locate(text, begin)
            if finish:
                _, right = locate(text, finish, after=left)
            else:
                right = text.find('\n\n', begin_end)
                if right < 0:
                    right = len(text)
            quote = text[left:right]
            spans.append({'paper_id': pid, 'document_sha256': source['sha256'],
                          'title': source['title'], 'source_url': catalog[pid].get('landing_url', ''),
                          'pdf_page': page, 'char_start': left, 'char_end': right,
                          'quote': quote, 'quote_sha256': hashlib.sha256(quote.encode()).hexdigest()})
        facts[key] = {'unit_id': key, 'answer_point': point, 'alternatives': [spans]}

    rows = []

    def Q(qid, zh, en, keys, category='single_paper', scope='named_papers', notes=None):
        units = [facts[key] for key in keys]
        pids = sorted({s['paper_id'] for u in units for alt in u['alternatives'] for s in alt})
        splits = {'dev' if p in DEV else 'heldout' for p in pids}
        assert len(splits) == 1, (qid, 'cross-split source leak')
        rows.append({'question_id': qid, 'split': splits.pop(), 'category': category,
                     'question': zh, 'question_en': en, 'language': 'zh', 'answerable': True,
                     'source_scope': scope, 'gold_paper_ids': pids,
                     'gold_sources_exhaustive': scope == 'named_papers',
                     'evidence_spans_exhaustive': False,
                     'reference_answer': ''.join(u['answer_point'] for u in units),
                     'required_answer_points': [{'point_id': u['unit_id'], 'text': u['answer_point'],
                                                 'evidence_unit_ids': [u['unit_id']]} for u in units],
                     'evidence_units': units,
                     'annotation': {'method': 'agent_source_grounded', 'human_verified': False,
                                    'retrieval_used_to_select_gold': False,
                                    'status': 'complete_source_checked',
                                    'notes': notes or []}})

    Q('paper001', '根据 RAGAs 原论文，Faithfulness 指标如何从生成回答和检索上下文计算出来？请说明步骤与分数定义。',
      'According to the original RAGAs paper, how is Faithfulness computed from the generated answer and retrieved context? Explain the steps and score definition.',
      ['r_statements', 'r_verify', 'r_fraction'], notes=['此前已用于检索调试，只可作为开发样本。'])
    Q('paper002', 'RAGAs 原论文如何计算 Answer Relevance？这个指标能否直接证明回答事实正确？',
      'How does the original RAGAs paper compute Answer Relevance? Does this metric directly establish factual correctness?',
      ['r_relevance', 'r_questions'])
    Q('paper003', 'ARES 开始评估前需要准备哪些输入？人工偏好标注和少样本问答有什么数量要求？',
      'What inputs does ARES require, including the approximate amount of human preference validation data and few-shot query-answer examples?', ['a_inputs'])
    Q('paper004', 'ARES 为什么在评判器之外还需要 PPI？人工验证集在其中起什么作用？',
      'Why does ARES use prediction-powered inference in addition to judges, and what role does its human validation set play?', ['a_pipeline', 'a_ppi'])
    Q('paper005', 'Active Retrieval Augmented Generation 论文中的 FLAREdirect 如何决定是否检索以及何时重新生成？',
      'How does FLAREdirect in Active Retrieval Augmented Generation decide whether to retrieve and when to regenerate?', ['f_trigger'])
    Q('paper006', 'FLAREdirect 如何避免把临时句中的低置信度错误带进检索查询？请说明两种查询构造方式。',
      'How does FLAREdirect avoid propagating low-confidence errors in its temporary sentence into retrieval queries? Describe both query formulation methods.', ['f_query'])
    Q('paper007', 'IRCoT 如何让推理与检索交替进行？每一轮追加检索使用什么作为查询？',
      'How does IRCoT interleave reasoning and retrieval, and what does it use as the query for additional retrieval?', ['i_loop'])
    Q('paper008', 'IRCoT 在什么条件下停止循环？停止后如何使用已检索的信息？',
      'When does IRCoT stop its loop, and how is the collected information used afterward?', ['i_stop'])
    Q('paper009', 'DPR 比较了哪三类负例？最佳方案如何组合 batch 内负例与 BM25 负例？',
      'Which three negative passage types does DPR compare, and how does its best model combine in-batch and BM25 negatives?', ['d_negatives'])
    Q('paper010', 'DPR 的 in-batch negatives 如何形成 B×B 分数矩阵？正例、负例如何区分？',
      'How do DPR in-batch negatives form a B by B score matrix, and which pairs are positive or negative?', ['d_batch'])
    Q('paper011', 'RocketQAv2 的动态 listwise 蒸馏如何处理检索器与重排器？是否冻结其中一个？',
      'How does RocketQAv2 dynamic listwise distillation handle the retriever and reranker? Is either module frozen?', ['rq_joint'])
    Q('paper012', 'RocketQAv2 描述的 cross-encoder 如何对 query-passage 打分？相比 dual-encoder 有什么能力和计算代价？',
      'How does the cross-encoder described in RocketQAv2 score a query-passage pair, and how does it compare with a dual-encoder in interaction modeling and computational cost?', ['rq_ce'])
    Q('paper013', 'Query Rewriting 论文的 Rewrite-Retrieve-Read 可训练方案中，哪些模块被冻结？改写器使用什么训练信号？',
      'In the trainable Rewrite-Retrieve-Read framework in Query Rewriting, which modules are frozen and what signal trains the rewriter?', ['w_reward'])
    Q('paper014', 'Query Rewriting 论文为什么选择互联网搜索引擎作为检索器？',
      'Why does the Query Rewriting paper use an internet search engine as its retriever?', ['w_web'])
    Q('paper015', 'RaFe 训练查询改写器的反馈是什么？支持哪些反馈训练形式？',
      'What feedback does RaFe use to train its query rewriter, and which feedback training modes does it support?', ['ra_reward'])
    Q('paper016', 'RaFe 在使用排序反馈前是否需要冷启动？初始改写训练数据从哪里来？',
      'Does RaFe require a cold start before ranking feedback, and where does its initial rewrite training data come from?', ['ra_cold'])
    Q('paper017', 'QASPER 提问者可以读到论文的哪些内容？问题与答案由同一组人提供吗？数据规模是多少？',
      'What parts of a paper can QASPER question writers read? Are questions and answers provided by the same group, and what is the dataset size?', ['q_collection'])
    Q('paper018', 'QASPER 需要哪些类型的证据？多段证据与图表证据分别占多少？原论文文本实验是否覆盖图表问题？',
      'What evidence types does QASPER require, what fractions need multiple paragraphs or figures/tables, and do its original text-only experiments cover figure/table questions?', ['q_evidence', 'q_text_scope'])
    Q('paper019', 'SciDQA 的问题和答案分别来自哪里？数据规模是多少？',
      'Where do SciDQA questions and answers originate, and how many QA pairs does the dataset contain?', ['sci_source'])
    Q('paper020', 'SciDQA 是否为全部问答标注了证据？是否存在需要读取被引用论文的问题？',
      'Does SciDQA annotate evidence for every QA pair, and does it include questions requiring explicitly cited papers?', ['sci_subset', 'sci_multidoc'])
    Q('paper021', 'LightRAG 的低层与高层检索分别关注什么？向量检索与图邻居如何配合？',
      'What do low-level and high-level retrieval focus on in LightRAG, and how do vector matching and graph neighbors work together?', ['l_levels', 'l_matching'])
    Q('paper022', 'LightRAG 如何把新文档加入已有图索引？是否需要从头重建整个索引？',
      'How does LightRAG incorporate new documents into its graph index? Does it require rebuilding the entire index?', ['l_update'])
    Q('paper023', 'GRAG 为什么采用先检索 ego-graphs 再合并与软剪枝的方法？',
      'Why does GRAG retrieve ego-graphs before merging and soft pruning them?', ['g_retrieve'])
    Q('paper024', 'GRAG 如何通过 hard prompts 与 soft prompts 同时保留文本和图拓扑信息？',
      'How does GRAG preserve both textual and topological information using hard and soft prompts?', ['g_prompt'])
    Q('paper025', 'LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering 中的两种信息视角是什么？分别由哪个模块处理？',
      'In LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering, what are its two information perspectives and which modules handle them?', ['long_dual'], notes=['用完整标题区分同名 LongRAG 论文。'])
    Q('paper026', 'LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering 为什么将短检索块映射回源长段落？',
      'Why does LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering map retrieved short chunks back to source long paragraphs?', ['long_map'])
    Q('paper027', 'SCITAT 有多少问题与来源论文？四种主要推理类型是什么？',
      'How many questions and source papers does SCITAT contain, and what are its four main reasoning types?', ['t_types'])
    Q('paper028', 'SCITAT 的 CAR 基线中 Calculator 和 Reasoner 如何协作？',
      'How do Calculator and Reasoner cooperate in the CAR baseline for SCITAT?', ['t_car'])

    seed_path = ROOT / 'data/evaluation/cross_paper_seed.json'
    seed = json.loads(seed_path.read_text(encoding='utf-8'))
    cross_specs = [
        ('Compare the evaluation aspects of RAGAs and ARES. Do they both require no human annotations at all? Explain the conditions.',
         ['r_dimensions', 'r_reference_free', 'r_humans', 'a_inputs', 'a_dimensions', 'a_pipeline']),
        ('When do FLARE and IRCoT retrieve again, and how do their triggering mechanisms differ?', ['f_trigger', 'f_instruct', 'i_loop', 'i_stop']),
        ('How do Query Rewriting and RaFe learn or optimize query rewriting, and what are their optimization signals?', ['w_reward', 'ra_reward', 'ra_cold']),
        ('How do DPR and RocketQAv2 differ in retriever and reranker training?', ['d_focus', 'd_arch', 'd_negatives', 'rq_joint']),
        ('How do QASPER and SciDQA differ in question sources and evidence annotation?', ['q_collection', 'q_evidence', 'sci_source', 'sci_subset']),
        ('Do LightRAG and GRAG retrieve the same graph objects and organize information for generation in the same way?',
         ['l_levels', 'l_matching', 'l_answer', 'g_retrieve', 'g_prompt']),
        ('Can the papers\' reported scores alone establish that LightRAG universally outperforms LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering? Examine datasets, models, and metrics.',
         ['l_setup', 'l_model', 'long_setup']),
        ('Why does SCITAT require scientific tables and text together, and can QASPER\'s original text-only evaluation fully cover that ability?', ['t_need', 'q_evidence', 'q_text_scope']),
    ]
    for item, (en, keys) in zip(seed['questions'], cross_specs, strict=True):
        Q(item['id'], item['question'], en, keys, category='cross_paper')
    next(r for r in rows if r['question_id'] == 'cross07')['reference_answer'] += '不能据此断言全面优于；需在相同数据、模型、预算和指标下作受控比较。'
    next(r for r in rows if r['question_id'] == 'cross07')['required_answer_points'].append({
        'point_id': 'comparison_limit', 'text': '实验条件与指标不同，不能直接比较分数或推出普遍优势。',
        'evidence_unit_ids': ['l_setup', 'l_model', 'long_setup'], 'reasoning': 'comparison_of_reported_conditions'})

    Q('open01', '有没有先把生成回答拆成陈述，再检查每个陈述是否被检索上下文支持的 RAG 评估方法？它的分数如何定义？',
      'Which RAG evaluation method first decomposes a generated answer into statements and checks whether the retrieved context supports each statement? How is its score defined?',
      ['r_statements', 'r_verify', 'r_fraction'], scope='open_corpus_anchor', notes=['未穷举 500 篇论文中的所有等价来源；只报告已标注证据覆盖率。'])
    Q('open02', '如何在一个训练 batch 内复用其他问题的正段落作为负例？问题与段落的分数矩阵如何构造？',
      'How can positive passages for other questions be reused as negatives in one training batch, and how is the query-passage score matrix formed?',
      ['d_batch'], scope='open_corpus_anchor')
    Q('open03', '有没有根据生成的下一句是否低置信度来决定再次检索、随后重新生成这句话的方法？请说明触发条件。',
      'Is there a method that triggers retrieval based on low confidence in a temporary next sentence and then regenerates the sentence? Explain the trigger.',
      ['f_trigger'], scope='open_corpus_anchor')
    Q('open04', '多跳问答中，能否交替用已有证据扩展推理，再用最新推理句寻找下一批证据？什么时候停止？',
      'In multi-hop QA, can retrieval and reasoning alternate, using the latest reasoning sentence to find more evidence? When does the loop stop?',
      ['i_loop', 'i_stop'], scope='open_corpus_anchor')

    def N(qid, pid, zh, en, missing):
        document_text, _ = folded('\n'.join(p['text'] for p in sources[pid]['pages']))
        assert folded(missing)[0] not in document_text
        rows.append({'question_id': qid, 'split': 'dev' if pid in DEV else 'heldout',
                     'category': 'unsupported', 'question': zh, 'question_en': en, 'language': 'zh',
                     'answerable': False, 'source_scope': 'named_papers', 'gold_paper_ids': [pid],
                     'gold_sources_exhaustive': True, 'reference_answer': '指定论文未报告该项结果，无法依据该论文给出所请求的数值。',
                     'evidence_spans_exhaustive': False,
                     'required_answer_points': [{'point_id': 'insufficient_information',
                         'text': '说明指定论文未提供这一结果，不编造数值。', 'evidence_unit_ids': []}],
                     'evidence_units': [], 'absence_check': {'paper_id': pid,
                         'document_sha256': sources[pid]['sha256'], 'pages_checked': len(sources[pid]['pages']),
                         'absent_term': missing, 'method': 'full_text_scope_check',
                         'scope_note': '只针对指定论文缺失该项结果；不代表全论文库不可回答。'},
                     'annotation': {'method': 'agent_source_grounded', 'human_verified': False,
                                    'retrieval_used_to_select_gold': False, 'status': 'complete_source_checked', 'notes': []}})
    N('negative01', RAGAS, 'RAGAs 原论文中，DeepSeek V4 Pro 的 Faithfulness 结果是多少？请只根据这篇原论文回答。',
      'What Faithfulness result does the original RAGAs paper report for DeepSeek V4 Pro? Answer only from that paper.', 'DeepSeek')
    N('negative02', DPR, 'DPR 原论文报告在 RTX 5090 上编码 100 万段落耗时多少秒？请只依据该论文给出数值。',
      'How many seconds does the original DPR paper report for encoding one million passages on an RTX 5090? Answer only from that paper.', '5090')
    N('negative03', SCITAT, 'SCITAT 原论文报告 DeepSeek V4 Pro 的 Exact Match 是多少？请只根据原论文回答。',
      'What Exact Match does the original SCITAT paper report for DeepSeek V4 Pro? Answer only from that paper.', 'DeepSeek')
    N('negative04', REWRITE, 'Query Rewriting 原论文使用 DeepSeek V4 Pro 阅读器时，改写器相对不改写提升了多少 F1？只根据该原论文回答。',
      'How much F1 improvement from rewriting does the original Query Rewriting paper report with a DeepSeek V4 Pro reader? Answer only from that paper.', 'DeepSeek')

    output = ROOT / 'data/evaluation/paper_benchmark'
    for split in ['dev', 'heldout']:
        subset = [r for r in rows if r['split'] == split]
        atomic_text(output / f'{split}.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset))
    atomic_json(output / 'manifest.json', {
        'schema_version': '1.0', 'benchmark_version': 'paper-grounded-v1',
        'annotation_method': 'agent_source_grounded', 'human_verified': False,
        'corpus_target': '500-paper global retrieval; do not filter by gold_paper_ids',
        'scope': 'All eight existing cross-paper seeds plus single-paper, unnamed-source and scoped unsupported questions; not exhaustive labeling of 31,037 chunks.',
        'gold_basis': 'PDF hash + normalized_blocks page text + exact character spans; chunk IDs are not gold.',
        'source_normalization': 'rag.chunking.normalized_blocks: per-block whitespace normalization, original block order, double newline between blocks. Offsets refer to these canonical page texts.',
        'evidence_policy': 'Source-checked sufficient evidence anchors, not an exhaustive enumeration of every valid passage or paraphrase; coverage is not universal relevance or answer accuracy.',
        'query_translation': 'Fixed agent-authored English equivalent for controlled cross-language experiments; keep constant across variants.',
        'split_policy': 'Disjoint gold paper groups including cross-paper links; RAGAs debugging samples kept in dev.',
        'heldout_status': 'Annotated from source, no retrieval/tuning performed on this split during preparation. Small diagnostic set, not an official or independently human-validated benchmark.',
        'open_question_policy': 'Anchored evidence not exhaustive across corpus; do not interpret unmatched alternative evidence as false.',
        'unsupported_policy': 'No positive retrieval gold; evaluate grounded insufficiency separately from evidence recall.',
        'counts': {'total': len(rows), 'splits': dict(Counter(r['split'] for r in rows)),
                   'categories': dict(Counter(r['category'] for r in rows)),
                   'answerable': sum(r['answerable'] for r in rows), 'source_papers': len(sources)},
        'source_documents': {pid: {'title': value['title'], 'sha256': value['sha256']} for pid, value in sources.items()},
    })
    # Preserve question IDs and original question wording; point to fully annotated rows.
    seed['status'] = 'complete_agent_source_grounded_not_human_verified'
    seed['annotation_dataset'] = 'data/evaluation/paper_benchmark/{split}.jsonl'
    by_id = {r['question_id']: r for r in rows}
    for item in seed['questions']:
        row = by_id[item['id']]
        assert set(item['paper_ids']) == set(row['gold_paper_ids'])
        item['annotation_question_id'] = row['question_id']
        item['annotation_split'] = row['split']
    atomic_json(seed_path, seed)
    print(json.dumps({'questions': len(rows), 'splits': dict(Counter(r['split'] for r in rows)),
                      'evidence_units': len(facts)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
