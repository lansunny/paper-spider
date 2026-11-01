# 评测说明

## 数据集与评分对象

| 数据集 | 题目 | 构成 | 来源 |
| --- | ---: | --- | --- |
| 文本 test50 | 50 | 46 道可回答、4 道证据不足问题；16 篇来源论文 | [题目](../data/evaluation/paper_benchmark/test50.jsonl) · [清单](../data/evaluation/paper_benchmark/test50_manifest.json) |
| 公式 test50 | 50 | 35 道语义、8 道编号、3 道字面符号、4 道控制题；19 篇论文、35 个原文公式位置 | [题目](../data/evaluation/formula_test50/questions.jsonl) · [来源标注](../data/evaluation/formula_test50/curation.json) |

文本证据以 PDF 哈希、物理页码、规范字符区间与原文片段定义。公式增加原始 PDF 几何位置。改变分块或抽取 ID 不会自动改变标准答案。

标注由源文档核对流程生成，尚未完成独立人工复核。开放式问题的有效证据并非穷尽；同一公式在其他位置重复出现时，严格位置评分可能不接受等价表达。公式题从已检测候选中选择，不能据此推断检测器召回率。

文本 50 题包含 23 道已评估问题与 27 道新增问题，结果按两组分别保留。多题共享论文与证据，不是 50 个完全独立样本。参与过调参的问题不应继续作为全新盲测结果。

## 指标定义

**证据单元平均召回率**：先对每题计算已覆盖必要证据单元数 / 必要证据单元总数，再按题求平均。一个单元可能包含多个必须共同覆盖的原文片段；多个可接受来源之间用 OR 表示，重叠文本不重复计分。

**公式锚点平均召回率**：每题已覆盖的必要公式组 / 所有必要公式组，再按题平均。组内允许替代来源，跨论文题要求覆盖各篇必要公式。匹配使用论文身份、PDF 哈希、页码与标注几何范围。

**全部目标齐全比例**：所有必要证据组都命中的问题数 / 可回答问题数。它比部分证据的平均覆盖更严格。

**控制项通过**：检查代码/表格标签误附带、缺失公式编号、无论文范围编号查找等检索行为。它不等于生成模型拒答准确率。

**来源重叠**：只检查文本块与公式源字符是否重叠，用于诊断；它既不证明公式结构完整，也不证明回答正确。

## 公式 V1 / V2 结果

两次完整检验使用相同 500 篇语料索引与同一套标注：

- snapshot：faa426834be428c99b05e883f7b5d303。
- questions SHA-256：b0efb852b74e452150774950d7ccfcf36f7f27be1da2ec3f4ad394e978b67853。
- gold SHA-256：978b185093265b56c852eff281d8c6c391c2336b2e370ab484191ab1fa4807ce。

| 条件 | V1 | V2 |
| --- | --- | --- |
| BM25 查询 | 固定英文等义问题 | 原始中文问题 + 科研术语扩展 |
| 按论文子查询 | 关闭 | 开启 |
| 跨论文候选名额与公式来源保护 | 关闭 | 开启 |
| 正文评分 | 标题 + 正文 | 原问题 + 对应论文子问题 |
| 公式排序 | 来源插入顺序 | 本地 Cross-Encoder，截断前排序 |
| 上下文预算 | 8,000 字符 | 16,000 字符 |
| 正文候选 | 20 条 | 20 条；评分对数可能更多 |
| 生成回答 | 关闭 | 关闭 |

这是整体配置比较，多个变量同时变化；提升不能被归因于某一个独立模块。

| 题型 | 题数 | V1 最终公式锚点平均召回 | V2 最终公式锚点平均召回 | V2 全部目标齐全 |
| --- | ---: | ---: | ---: | ---: |
| 单论文语义查询 | 20 | 80.0% | 85.0% | 85.0% |
| 开放式语义查询 | 10 | 60.0% | 70.0% | 70.0% |
| 跨论文语义查询 | 5 | 0.0% | 60.0% | 40.0% |
| 语义题合计 | 35 | 62.9% | 77.1% | 74.3% |
| 编号精确查询 | 8 | 100.0% | 100.0% | 100.0% |
| 字面符号查询 | 3 | 100.0% | 100.0% | 100.0% |

控制项：V1 为 3/4，V2 为 4/4。跨论文全部目标齐全仍只有 2/5，是当前主要改进方向。

公开记录：[V1 摘要](benchmarks/formula-v1.summary.json)、[V2 摘要](benchmarks/formula-v2.summary.json)、[V1 配置](benchmarks/formula-v1.profile.json)、[V2 配置](benchmarks/formula-v2.profile.json)。

## 文本 V1 基线

BM25 使用固定英文等义问题；Dense 与重排使用中文问题；固定窗口语料；每路 100 条、RRF 60、20 条正文重排候选、8,000 字符上下文预算。

| 阶段 | 必要证据平均覆盖 | 所有证据齐全的问题比例 |
| --- | ---: | ---: |
| 双路候选池 | 92.3% | 87.0% |
| 融合前 20 条 | 79.5% | 69.6% |
| 重排前 5 条 | 67.4% | 58.7% |
| 最终预算上下文 | 59.4% | 50.0% |

这是旧版文本基线，不能与新版公式检验的统计对象混用。具体分组见 [文本摘要](benchmarks/text-v1.summary.json)。

## 重跑与消融

先完成 [语料、模型、索引与公式产物准备](getting-started.md)，再生成本地参考裁剪：

~~~bash
python scripts/prepare_formula_test50.py
python scripts/test_formula_retrieval50.py --validate-only
python scripts/test_formula_retrieval50.py --profile v2 --limit 3
python scripts/test_formula_retrieval50.py --profile v2
~~~

独立比较某个机制时，固定语料、题目、查询语言和预算：

~~~bash
python scripts/test_formula_retrieval50.py --profile v2 --context-budget 16000
python scripts/test_formula_retrieval50.py --profile v2 --context-budget 16000 --no-decomposition
python scripts/test_formula_retrieval50.py --profile v2 --context-budget 16000 --no-lexical-expansion
python scripts/test_formula_retrieval50.py --profile v2 --context-budget 16000 --no-formula-rescue
python scripts/test_formula_retrieval50.py --profile v2 --context-budget 16000 --no-formula-rank
~~~

这些关闭开关在当前实现中可能存在依赖：关闭拆解也会使依赖论文子问题的机制不再执行；关闭公式排序会连带停用依赖其评分的公式来源保护。比较的是实际开关组合，不能假定每个开关都严格正交。

命令行 V1 为历史英文辅助基线；V2 默认使用中文问题。若比较其他查询输入，应显式说明英文辅助条件。文本历史入口为 accept_heldout.py 与 accept_test50.py，结果锁定由 frozen_baseline.py 管理，完整 50 题入口会复用先前 23 题验收记录。

## 复现范围

摘要是已完成运行的记录；新克隆仓库需要重建本地原文、参考裁剪、模型缓存和索引。源码、配置和数据哈希指纹随结果保存。设备、依赖、近似检索和源 PDF 版本变化都可能影响结果。

这些检验不包含端到端回答正确性、断言级引用支持、视觉公式理解、独立 OCR 准确率或线上并发性能。

回到 [项目首页](../README.md)。

