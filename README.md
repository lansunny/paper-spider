<p align="center">
  <picture>
    <source media="(max-width: 640px)" srcset="docs/assets/cover-mobile.svg">
    <img src="docs/assets/cover.svg" alt="PaperRAG — 让论文回答，有原文可循。混合检索、跨论文问答与公式原图溯源。" width="100%">
  </picture>
</p>
<p align="center"><strong>面向科研论文的证据检索工作台，支持跨论文问答与公式原图溯源。</strong></p>
<p align="center">A paper research workspace for hybrid retrieval, cross-paper comparison, and source-linked formulas.</p>
<p align="center">
  <img src="docs/assets/badges/python.svg" alt="Python 3.12" height="25">
  <img src="docs/assets/badges/retrieval.svg" alt="Hybrid retrieval: BM25 + BGE-M3" height="25">
  <img src="docs/assets/badges/qdrant.svg" alt="Vector database: Qdrant" height="25">
  <a href="LICENSE"><img src="docs/assets/badges/license.svg" alt="License: Apache-2.0" height="25"></a>
</p>
<p align="center">
  <a href="#界面预览">界面预览</a> ·
  <a href="#项目特点">项目特点</a> ·
  <a href="#实验结果">实验结果</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="docs/architecture.md">技术设计</a> ·
  <a href="docs/evaluation.md">评测说明</a>
</p>

## 从研究问题，到可核对的原文与公式

PaperRAG 将中文研究问题连接到英文论文中的原文段落与公式位置。单论文问题限定检索范围，跨论文问题按论文组织证据，回答附带可跳转的来源引用；公式通过正文坐标联接，直接呈现原始 PDF 裁剪。召回、融合、重排和上下文选择均保留阶段记录，便于分析证据遗漏发生在哪一步。

<table>
<tr>
<td width="33%"><strong>500 篇论文</strong><br>RAG、检索、评估与长上下文相关文献</td>
<td width="34%"><strong>31,037 个文本块</strong><br>论文身份、PDF 页码与来源信息</td>
<td width="33%"><strong>4,473 个公式候选</strong><br>覆盖 8,624 页的规则抽取结果</td>
</tr>
</table>

以上为已完成实验的语料规模。仓库提供代码、来源清单和评测标注；论文 PDF、模型权重与索引在本地恢复和构建。

## 界面预览

**文献、问题与原文阅读台。** 左侧限定检索范围，中间组织问题与结果，右侧核对引用对应的论文页面。

![PaperRAG 论文研究工作台](docs/assets/screenshots/workspace.jpg)

<table>
<tr>
<td width="74%" valign="top">
<strong>公式原图与关联正文</strong><br><br>
<img src="docs/assets/screenshots/formulas.jpg" alt="公式编号检索、原始裁剪与来源正文联接" width="100%">
</td>
<td width="26%" valign="top">
<strong>移动端研究界面</strong><br><br>
<img src="docs/assets/screenshots/mobile.jpg" alt="移动端问题输入与文献阅读" width="100%">
</td>
</tr>
</table>

## 项目特点

### 双路召回与细粒度重排

- **BM25 + BGE-M3** 并列召回，以词项匹配和多语言语义表示互补；当前使用 BGE-M3 的 1024 维 dense 表示。
- **RRF 排名融合**整合两路候选，由 BGE-Reranker-v2-M3 对问题与「论文标题 + 正文」联合评分。
- 中文查询采用 **bigram 分词与科研术语中英扩展**；候选来源、排序分数与阶段耗时可追踪。

### 按论文组织跨文献证据

- 单篇且无歧义的问题，通过论文 ID 限定检索范围。
- 明确提及多篇论文时，保留原问题检索，并构造按论文约束的子查询，为各篇论文预留候选名额。
- 全部重排候选参与 **16,000 字符上下文预算**筛选；Top5 保留为排序诊断口径。

### 正文与公式共享来源坐标

- 利用共同字符坐标，联接块内公式、相邻说明及显式公式引用，**复用文本索引完成公式检索**。
- 补充单论文范围内的编号精确查找和符号文字匹配，并在展示截断前排序公式候选。
- **原图用于呈现，文字代理用于检索**。裁剪来自原始 PDF，通过文档哈希、页码和坐标验证来源一致性。
- 完整文档的权威抽取记录作为唯一入口，避免多次抽取产物混用。

### 面向失败阶段的评测

- 文本与公式各有一套 **50 题检验集**，覆盖单论文、跨论文、开放式问题及证据不足或异常查询。
- 标准证据锚定原始 PDF、页码及字符范围；公式标注还保留几何位置，减少分块变化对标注的影响。
- 分别测量候选池、重排候选、Top5 与预算上下文中的证据覆盖，定位召回、截断、排序和上下文选择的损失。

## 系统流程

![文档建库、查询检索与公式来源联接](docs/assets/architecture.svg)

| 层次 | 实现 |
| --- | --- |
| PDF 与来源 | PyMuPDF；物理 PDF 页码、正文位置、文档 SHA-256 |
| 分块 | 固定 token 窗口；另有独立的版面结构分块实验配置 |
| 向量存储 | Qdrant；命名向量、论文过滤与 snapshot 隔离 |
| 检索与排序 | BM25、BGE-M3 dense、RRF、标题增强 Cross-Encoder |
| 问题处理 | 保守论文路由、按明确论文实体拆解、候选配额 |
| 公式 | 规则检测、原图裁剪、共享坐标联接、编号查找与候选排序 |
| 回答 | DeepSeek API 适配；证据预算、引用编号与清单一致性检查 |
| 工作台 | FastAPI + 原生 HTML/CSS/JavaScript；本地字体与响应式阅读台 |

运行主体位于 [rag/](rag/)、[backend/](backend/) 和 [frontend/](frontend/)。实现细节见 [技术设计](docs/architecture.md)。

## 实验结果

**公式检索：同一套 50 题检验集，其中 35 题为语义查询。**

| 指标 | V1 | V2 |
| --- | ---: | ---: |
| 35 道语义题：最终上下文公式锚点平均召回率 | 62.9% | **77.1%** |
| 35 道语义题：全部目标公式齐全的问题比例 | 62.9% | **74.3%** |
| 5 道跨论文题：公式锚点平均召回率 | 0.0% | **60.0%** |
| 编号 / 符号查询：全部目标齐全 | 11 / 11 | 11 / 11 |
| 检索控制项通过 | 3 / 4 | **4 / 4** |

“公式锚点召回”衡量是否找回标注的原文公式位置；“全部目标齐全”要求一道题的每组必要公式都被覆盖。V1 与 V2 是整体配置对照，具体差异与分题型结果见 [评测说明](docs/evaluation.md)。

<details>
<summary><strong>文本检索基线：候选覆盖与最终上下文覆盖</strong></summary>

50 题中 46 题可回答。以下为固定窗口 V1 基线，BM25 使用固定英文等义查询，预算为 8,000 个序列化证据字符。

| 阶段 | 标注证据单元平均召回率 |
| --- | ---: |
| 双路候选池 | **92.3%** |
| 融合后前 20 条 | 79.5% |
| 重排前 5 条 | 67.4% |
| 预算内最终上下文 | 59.4% |

候选覆盖与最终上下文覆盖之间的差距，是后续候选保留与上下文策略的优化依据。

</details>

结果摘要与配置指纹见 [docs/benchmarks/](docs/benchmarks/)。这些指标衡量证据检索与原文位置覆盖，不等同于问答准确率或完整公式检测召回率。

## 快速开始

推荐 Python 3.12。下列命令使用已验证的 CUDA 12.4 依赖配置；CPU 与 Linux 安装方式见 [运行指南](docs/getting-started.md)。

在仓库根目录执行：

~~~powershell
python -m venv .venv-rag
.\.venv-rag\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 启动本地 Qdrant
docker compose up -d qdrant

# 按固定清单恢复原始 PDF，校验 SHA-256
python scripts/restore_paper_corpus.py --download

# 下载固定版本模型，建立向量索引
python main.py --task models --config configs/retrieval_v2.json
python main.py --task index --config configs/retrieval_v2.json

# 抽取全文公式并登记权威产物
python scripts/extract_all_formulas.py

# 启动工作台
python main.py
~~~

打开 **http://127.0.0.1:8765**。首次模型准备、全量 PDF 恢复和向量化需要下载与计算时间。

如需生成回答，将 [.env.example](.env.example) 复制为 <code>.env</code>，在本地填写 <code>DEEPSEEK_API_KEY</code>。模型 ID 可通过 <code>DEEPSEEK_MODEL_ID</code> 调整；检索与公式抽取本身不需要生成 API 密钥。

### 检验与消融

恢复语料、模型、索引和公式产物后，生成公式检验集的本地参考裁剪：

~~~powershell
python scripts/prepare_formula_test50.py
python scripts/test_formula_retrieval50.py --validate-only

# 小样本检查与完整检验
python scripts/test_formula_retrieval50.py --profile v2 --limit 3
python scripts/test_formula_retrieval50.py --profile v2

# 独立关闭机制，观察阶段覆盖与耗时
python scripts/test_formula_retrieval50.py --profile v2 --no-decomposition
python scripts/test_formula_retrieval50.py --profile v2 --no-formula-rescue
~~~

检验脚本不生成回答；V2 默认查询拆解、词汇扩展和排序在本地完成。比较条件见 [评测说明](docs/evaluation.md)。

## 代码入口

~~~text
PaperRAG/
├── rag/                     # 解析、分块、检索、重排、生成与公式来源联接
│   ├── pipeline_v2.py       # 当前工作台检索编排
│   ├── query_planning_v2.py # 论文路由与按论文子查询
│   ├── lexical_v2.py        # bigram 与科研术语扩展
│   ├── vector_store.py      # Qdrant、过滤条件与 snapshot
│   ├── formula_retrieval_v2.py
│   └── formula_store.py     # 权威公式产物与完整性验证
├── backend/                 # FastAPI、入库任务与 PDF/公式图片服务
├── frontend/                # 三栏研究工作台与移动端阅读
├── configs/                 # V1、V2 与结构分块独立配置
├── data/                    # 论文来源清单与固定评测标注
├── scripts/                 # 语料恢复、建库、抽取、检验和消融
├── tests/                   # 离线核心单元测试
├── docs/                    # 技术设计、评测口径、运行指南与界面预览
└── main.py                  # 本地启动与 CLI 入口
~~~

~~~powershell
python -m unittest discover -s tests -v
~~~

常规单元测试使用合成样例与模拟接口，不下载模型、不调用生成 API。真实 Qdrant 集成测试通过显式环境变量单独启用。

## 数据与许可

论文清单保留原始作者、来源 URL 与文件哈希。原文、模型、字体及依赖分别遵循各自许可；参考裁剪由本地原始 PDF 生成。详情见 [数据说明](docs/data.md) 与 [第三方声明](THIRD_PARTY_NOTICES.md)。

项目源代码采用 [Apache-2.0](LICENSE) 许可。

