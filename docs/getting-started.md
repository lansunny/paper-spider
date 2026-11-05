# 运行指南

## 运行条件

- Python 3.12。
- Docker Compose 与本地 Qdrant，默认地址为 http://127.0.0.1:6333。
- 可访问 ACL Anthology 与 Hugging Face，以恢复原文和固定版本模型。
- 可选的 DeepSeek API 密钥，用于生成回答。

requirements.txt 固定 PyTorch CUDA 12.4 构建。项目支持 CPU、CUDA 与自动选择设备。

## 安装依赖

Windows / NVIDIA：

~~~powershell
python -m venv .venv-rag
.\.venv-rag\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

Linux / CPU：

~~~bash
python3.12 -m venv .venv-rag
source .venv-rag/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-base.txt
~~~

requirements-base.txt 包含共享依赖；先安装适合平台的 PyTorch，再安装共享部分。当前发布包未在所有操作系统上执行完整建库与模型推理验证。

## 恢复固定语料

~~~bash
python scripts/restore_paper_corpus.py
python scripts/restore_paper_corpus.py --download
~~~

第一条只展示选择范围；第二条从 data/paper_catalog.json 的原始 URL 恢复 500 篇 PDF，校验记录的 SHA-256，不改写清单和标注。校验通过的现有文件直接复用。

可用 <code>--paper-id 2024.eacl-demo.16</code> 或 <code>--limit 5</code> 检查部分下载；固定清单的完整索引仍需要全部 PDF。现有文件哈希不同或官方文件发生变化时，脚本保留原文件并报错，避免改变证据来源。

## 模型与索引

~~~bash
docker compose up -d qdrant
python main.py --task models --config configs/retrieval_v2.json
python main.py --task index --config configs/retrieval_v2.json
~~~

模型保存在 runtime/models/。模型仓库 revision、向量维度和分块配置参与索引身份验证；更换表示模型或分块方式时使用匹配的独立索引。

纯 CPU 建库可在索引命令后追加 <code>--device cpu</code>。网页默认设备为 auto，会在 CUDA 可用时使用 GPU；也可将当前检索配置的 device 设置为 cpu。

retrieval.json 与 retrieval_v2.json 共享固定窗口的索引表示，主要区别为在线检索策略和上下文预算。结构分块使用 retrieval_structure.json 与独立 collection：

~~~bash
python scripts/prepare_structure_index.py --build
~~~

## 公式产物

~~~bash
python scripts/extract_all_formulas.py --dry-run
python scripts/extract_all_formulas.py
~~~

抽取使用原生 PDF 文本与版面，不调用 OCR 或生成 API。每篇完整抽取成功后登记到 runtime/formulas/CURRENT.json。同一配置下重跑会验证并复用缓存；局部页抽取不替换完整文档记录。

## 启动工作台

~~~bash
python main.py
~~~

访问 http://127.0.0.1:8765。更换端口使用 <code>python main.py --port 8770</code>。前端直接由 FastAPI 提供，无需 npm 构建；字体和图标随源码本地提供。

网页默认使用 V2。切换到冻结基线时，在启动进程环境中设置 <code>PAPERRAG_RETRIEVAL_PROFILE=v1</code>。

## 论文地图与研究综述

四维图默认选取 50 篇试点论文。先完成结构索引并设置本地 `.env`，再执行：

~~~bash
python scripts/prepare_structure_index.py --build
python scripts/build_paper_graph.py
python scripts/build_paper_graph.py --run
~~~

不带 `--run` 只预览；实际运行会通过配置的 DeepSeek 兼容 API 抽取主题、方法、实验和结论，再用本地 BGE-M3 编码，建立四张加权近邻图并运行 Leiden。该步骤会产生 API 用量，完成步骤可复用缓存。参数见 [paper_graph.json](../configs/paper_graph.json)，图产物保存在 `runtime/paper_graph/`，不随源码分发。

启动工作台后打开 `/graph` 浏览论文地图。打开 `/review`，在尚未准备社区卡片时点击准备入口；卡片制作会调用生成 API。卡片准备完成后，可以浏览群组、进入原文，或提交研究问题生成综述。界面保留任务进度和历史记录。

也可在终端只检查取证，不生成回答：

~~~bash
python scripts/research_review.py "RAG 评估有哪些研究路线？" --evidence-only
~~~

卡片需预先准备好。研究综述使用现有图覆盖的论文，不代表已覆盖完整 500 篇库。普通全文检索仍使用默认 V2 索引；新图产物不会自动改变其候选策略。

## 回答模型

复制 .env.example 为 .env，在本地填写密钥：

~~~dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL_ID=deepseek-v4-pro
DEEPSEEK_MAX_TOKENS=1500
~~~

可用模型以账户实际支持的 ID 为准。关闭「生成带引用的回答」后，可单独查看证据和公式原图。真实密钥不进入代码、截图或版本控制。

## 常见问题

| 现象 | 检查项 |
| --- | --- |
| 混合 / 语义检索不可用 | Qdrant 是否启动、地址和 collection 是否匹配 |
| 清单已存在但无法建库 | PDF 不随仓库分发，先完成固定语料恢复 |
| 检索正常但未生成回答 | 是否启用生成、密钥是否填写、模型 ID 是否可用 |
| 公式检验缺少参考图片 | 完成 PDF 恢复后运行 prepare_formula_test50.py |
| 结构配置找不到向量库 | 按独立配置建立结构索引，不能复用固定窗口 collection |
| 哈希或来源坐标不一致 | 保留现有标注，核对原 PDF 与抽取配置 |

回到 [项目首页](../README.md)。

