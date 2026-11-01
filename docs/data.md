# 数据与发布内容

## 固定文献清单

[paper_catalog.json](../data/paper_catalog.json) 与 [paper_manifest.json](../data/paper_manifest.json) 记录 500 篇论文的标题、作者、年份、ACL Anthology 页面、原 PDF URL 和来源哈希。主题包括 RAG、检索与重排、评估、长上下文、多跳推理及科学问答。

PDF 原文不随代码发布。恢复工具按固定清单取回官方文件并验证哈希；扩充语料的发现脚本保留为独立入口，不用于替换固定评测语料。

清单中的历史 downloaded 状态描述原实验时的文献准备状态，不表示新克隆仓库已经拥有这些 PDF。

## 公开标注

- 文本：开发、历史验收和 50 题检验标注，包含来源页码、字符范围、必要证据组与评测用参考答案。
- 公式：50 道问题与固定来源标注，包含 35 个公式原文位置。参考裁剪在恢复原 PDF 后本地生成。
- 实验：公开汇总结果、原运行配置与代码/数据指纹，见 [benchmarks/](benchmarks/)。

标准答案与来源标签只用于评分，不作为论文过滤条件，也不发送给回答生成器。标注未经过独立人工复核，且有效证据不是全语料穷尽标注。

## 本地生成内容

| 内容 | 默认位置 | 发布包 |
| --- | --- | --- |
| PDF 全文 | data/pdfs/ | 不包含 |
| 下载页面与 QASPER 原始/准备数据 | data/source_pages/ 等 | 不包含 |
| 模型权重 | runtime/models/ | 不包含 |
| 向量库与索引快照 | Docker volume、runtime/indexes/ | 不包含 |
| 公式原图与抽取缓存 | runtime/formulas/ | 不包含 |
| 公式参考裁剪 | data/evaluation/formula_test50/gold_images/ | 本地生成 |
| 上传文献 | runtime/uploads/ | 不包含 |
| 提问轨迹与本地运行日志 | runtime/ | 不包含 |
| 密钥与个人配置 | .env、编辑器配置 | 不包含 |
| 展示截图 | docs/assets/screenshots/ | 包含，经明确选择 |

数据与运行产物不通过代码发布包转移。恢复步骤见 [运行指南](getting-started.md)。

## 来源与许可

论文和第三方模型保留其原有权利与许可。来源页面中的许可记录并非每篇都已完成相同程度的核查，因此不将全部论文统一声明为可任意再分发。

展示截图中的 RAGAs 页面与公式片段保留来源说明，见 [第三方声明](../THIRD_PARTY_NOTICES.md)。代码许可证不替代文献、模型和依赖的许可条款。

回到 [项目首页](../README.md)。

