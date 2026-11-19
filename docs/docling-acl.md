# Docling ACL 检索库

使用当前 ACL 基线索引的论文清单和 PDF 哈希，另建 Qdrant 集合 `paperrag_docling_acl_<profile>`。Docling 环境与现有 RAG 环境独立，原有正文库、公式和 PDF 不修改。子集试验使用带 `_pilot` 后缀的集合。

解析阶段使用 Docling 的标准 PDF 流程、版面识别、表格结构与标题层级恢复。文本层正常的 ACL 论文默认关闭 OCR。保存完整 Docling JSON；每块保留元素 ID、类型、页码、区域框等 Docling 来源信息。Docling 的字符坐标不进入既有 `SourceSpan`；跨页块的 `pages` 包含全部可能来源页，`pdf_page` 只是首个来源页。重复表头、图注随对象结构保留；单独的图片检索块只索引已有图注，不对图片生成描述。复杂表格的行列归属仍须回 PDF 核查。

`HybridChunker` 使用本地 BGE-M3 tokenizer，默认切块目标为 384 token，含特殊符号的实际输入上限为 512。向量采用章节上下文增强后的文本，返回块正文和原始来源。改变 `--max-tokens` 会读取已有解析 JSON，再生成隔离的新切块版本。

在 PowerShell 中从项目目录运行：

```powershell
python -m venv runtime/docling-venv
$docling = '.\runtime\docling-venv\Scripts\python.exe'
& $docling -m pip install 'torch==2.6.0+cpu' 'torchvision==0.21.0+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
& $docling -m pip install 'docling-slim[standard]==2.150.1' 'docling-core==2.107.1' 'docling-parse==7.26.0' 'docling-ibm-models==4.0.4' 'transformers==4.57.6' 'pymupdf==1.26.5'
.\scripts\docling_acl.ps1 -Objects
```

也可把解析和建库分开运行：

```powershell
& .\runtime\docling-venv\Scripts\python.exe -u scripts/docling_prepare.py --threads 8
& .\.venv-rag\Scripts\python.exe -u scripts/docling_index.py build --profile <profile>
& .\.venv-rag\Scripts\python.exe scripts/docling_index.py search --profile <profile> 'RAGAs Faithfulness 如何计算'
```

解析按论文保存并校验产物哈希，重跑会跳过已完成论文。向量上传中断时，重复运行相同 profile 的 `build` 会续传缺失的向量。全量文档和向量数校验通过后才写入新集合的 `CURRENT.json`。解析与建库均不使用生成 API。

## 段落、原图与表格对象库

基于已缓存的 Docling PDF 解析结果生成独立对象 profile。每个正文元素单独成块，保留所属标题、PDF 哈希、页码、区域框、Docling 元素 ID 和元素内字符范围。超长元素按本地 BGE-M3 tokenizer 切成连续窗口；默认每个嵌入输入最多 384 token（含论文标题、章节路径和特殊符号），窗口目标 overlap 为 64 token，最后一个窗口保留段落尾部。短段落不合并。

SQLite 保存 `papers`、`objects`、`links` 和 `labels`。原始 Docling JSON 继续单独保存，PDF 区域裁图以 PNG 形式落盘。图片检索视图使用作者图注和区域内已有文本；表格保存原始单元格、跨度、表注、脚注和原页裁图，另生成表级及行级检索视图。网格校验失败的表保留原始结构和裁图，不生成行级描述；网格校验通过也不代表模型抽取的表格内容完全正确。表格 HTML 和行标注明确来自解析产物，必要时回原页核对。

正文按同篇论文内唯一的图表编号建立双向链接，记录编号在段落里的字符范围。重复编号或缺少编号的对象不自动连线；未匹配的引用留在对象的 `unresolved_mentions`。命中块带稳定 `object_id`，可以回取对象、文件与相关段落。已知论文时，`--paper-id` 支持图表编号精确定位。最终用户答复必须以论文原页为依据，勿把检索用的行描述误作原文摘录。

```powershell
$parseProfile = '9750eb159b572c55' # 当前 ACL Docling 解析配置
& .\runtime\docling-venv\Scripts\python.exe -m pip install pymupdf==1.26.5
.\scripts\docling_acl.ps1 -Profile $parseProfile -Objects
# 完成后，从 $parseProfile 下的 OBJECTS_BUILD_INPUT.json 读取新对象 profile。
& .\.venv-rag\Scripts\python.exe scripts/docling_index.py search --profile <object-profile> --paper-id 2023.acl-long.557 'Figure 2'
```

需要从头解析时可运行 `.\scripts\docling_acl.ps1 -Objects`。换窗口大小时，可在 `scripts/docling_objects.py` 传入 `--max-tokens` 和 `--overlap`，复用同一解析 profile。对照实验与线上默认库可保持并存。

解析仍在运行时，可以用 `scripts/docling_objects.py --parse-profile <profile> --cache-ready` 先重切已经解析完成的 PDF。该命令仅写逐篇缓存，不发布 SQLite 全库或 Qdrant 快照。

全库对象建库顺序为：完成切分与 SQLite 发布 → 全量来源校验 → 从同一对象快照抽取三篇论文到独立试点集合 → 验证正文混合检索、图表编号、裁图和正文链接 → 启动全量向量化。校验或试点出错会停止流水线。试点报告 `object-smoke.json` 和全量校验报告 `object-validation.json` 必须与待建库的对象快照一致。`--existing-pilot` 仅用于验证测试脚本，产生独立报告，不能放行全量建库。

## 多模态检索 API

在「多论文检索」选择「多模态检索」，可切换正文、图片和表格，或者上传 PNG、JPEG、WebP 进行以图搜图。论文栏可选多篇限定范围。检索结果支持预览图表、查看关联正文和跳转到 PDF 来源页。

- `GET /api/docling/profiles` 列出可用快照及 `clip_ready`。试点必须明确传 `profile` 和 `pilot: true`；缺少全量库时不会自动换成试点。
- `POST /api/docling/search` 传 `query`，可选 `paper_id`、`kinds`（`body`、`figure`、`table`）、`top_k`、英文视觉描述 `clip_query`。`visual: false` 关闭 CLIP；`nearby: false` 关闭几何近邻，保留明确编号链接。各类检索预览最多返回 `top_k` 条。
- 可用 `paper_ids` 数组限定多篇论文。图片上传使用同名 multipart 字段，每篇论文附一项。明确编号查询在单篇范围内进行精确定位。
- 请求包含正文时，系统重排 RRF 前 100 个正文块，按 16,000 字符预算生成 `body_context`。该字段携带引用编号和原文链接，实际占用见 `body_context_chars`。检索预览 `body` 仍由 `top_k` 控制。
- 正文、图注、表题和表格行在各自的候选范围内做 BM25、BGE-M3、RRF 和重排。图片按对象 ID 融合 `caption`、`context` 和 `clip` 的排名，表格按对象 ID 融合 `table`、`row`。同一对象在同一路中只计一次排名。
- Docling 默认用已配置的 DeepSeek 模型将中文问题译成英文。正文、图注、整表和表格行的 BM25、BGE-M3 与重排共享译文。调用方可用 `query_language: "original"` 保留原问，或用 `"bm25_english"` 仅在词项检索中使用译文。翻译失败时使用原问，并返回 `query_translation.status: "fallback"`。模型只接收问题文本。
- 论文已知时，单独查询 `Figure 2`、`Table 1` 直接查编号；附带分析要求时优先返回该对象并继续检索。正文邻近候选只考虑同篇同页且横向重叠的框，再按距离、章节和术语重叠排序；`nearby_candidate` 不等于论文里的明确引用。
- `POST /api/docling/search/image` 用 multipart 上传 `image`（PNG、JPEG 或 WebP），附 `profile`，试点附 `pilot=true`，可以限定 `paper_id`。CLIP 在本地编码文本或图片。文搜图默认使用译文，可用 `clip_query` 直接提供英文视觉描述。
- 结果的 `object_url` 可打开完整对象；`pdf_url` 和 `image_url` 分别指向 PDF 来源页及原页裁图。表格返回匹配行、单元格、表头和脚注，完整对象含全表和 HTML。`paths` 记录候选来自哪一路。`/api/ask` 的 `deep_research` 模式使用 Docling 正文。

```powershell
& .\.venv-rag\Scripts\python.exe scripts/build_docling_clip.py --download-model
$releasePath = Get-ChildItem .\runtime\docling-acl -Recurse -Filter OBJECTS_BUILD_INPUT.json |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$release = Get-Content -LiteralPath $releasePath -Raw | ConvertFrom-Json
& .\.venv-rag\Scripts\python.exe scripts/build_docling_clip.py --profile $release.profile_id --device cuda
```

图像向量旁库校验裁图哈希，并与 PDF 对象快照和文本索引快照绑定。缺少匹配的旁库时，普通文本查询仍可使用图注和上下文分支；直接上传图片搜索会提示旁库尚未就绪。

使用 `scripts/evaluate_docling_query_language.py` 在全量正文库上比较原问、仅 BM25 英译和全英文三条路径。扩展题来自旧文本评测中可在同版 PDF 的同页重新定位全部证据的单篇问题。评测记录逐题译文、候选及标注证据覆盖，翻译模型仅接收原问题。

使用 `scripts/evaluate_docling_text50.py` 在当前 Docling 正文检索上复跑历史固定窗口的全部 50 道题。用 `--body-rerank-k 20` 复现旧的重排深度，用 `--translations-from <已完成的评测目录>` 固定模型译文。报告包含同预算下的证据覆盖、上下文用量和无法完整映射到正文索引的题目。4 道证据不足题保存检索结果，拒答质量需通过回答评测核对。

项目的 CUDA 续跑入口为 `.\scripts\docling_acl.ps1 -Profile 9750eb159b572c55 -Threads 4 -Objects -CudaResume`。后台使用单独的 `runtime/docling-cuda-venv`；确认 CUDA 可用后，仅解析缺少有效缓存的论文。CUDA 解析失败则用原 CPU 环境继续，已完成 PDF 不重跑。CPU 和 GPU 使用同一解析配置；每篇新解析产物记录实际请求设备和 PyTorch 构建版本。切换处理设备前应停止旧解析任务，避免两个进程写同一缓存。
