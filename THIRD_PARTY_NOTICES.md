# Third-party notices

Paper spider 的运行主体位于 `rag/`、`backend/` 与 `frontend/`。依赖、模型、论文与字体分别适用其原始许可。

| 组件 | 用途 | 来源与许可说明 |
| --- | --- | --- |
| PyMuPDF | PDF 原生文本、版面坐标与原图裁剪 | [PyMuPDF](https://github.com/pymupdf/PyMuPDF)；AGPL-3.0 或商业授权，使用与分发须遵循适用条款 |
| rank-bm25 | 词法检索 | [rank-bm25](https://github.com/dorianbrown/rank_bm25)；Apache-2.0 |
| Qdrant | 向量存储与元数据过滤 | [Qdrant](https://github.com/qdrant/qdrant)；Apache-2.0 |
| BGE-M3 | 多语言稠密向量表示 | [模型卡](https://huggingface.co/BAAI/bge-m3)；以模型仓库许可为准 |
| BGE-Reranker-v2-M3 | 问题与候选联合评分 | [模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)；以模型仓库许可为准 |
| IBM Plex Sans | 前端字体 | [IBM Plex](https://github.com/IBM/plex)；SIL Open Font License 1.1，字体许可见 [OFL.txt](frontend/assets/fonts/OFL.txt) |
| kotaemon | 早期工程参考 | [kotaemon](https://github.com/Cinnamon/kotaemon)，参考提交 `9ad3e4e49aa35b8acddd235918a5d9753c1cfdf9`；该参考源码不随发布包分发 |
| taste-skill / interface-design | 界面设计参考 | [taste-skill](https://github.com/Leonxlnx/taste-skill)、[interface-design](https://github.com/Dammyjay93/interface-design)；不随发布包分发其工具源码 |
| ashi-liquid-glass | 原生 Canvas / SVG 玻璃折射、RGB 边缘色散与双侧高光 | [ashi-liquid-glass](https://github.com/the-Ashkanian/ashi-liquid-glass)；MIT，原始许可见 [LICENSE](frontend/assets/licenses/ashi-liquid-glass.LICENSE.txt) |
| Prompt Spider UI | 桌面悬浮蜘蛛的机械步态、线框身体与伸爪绘制；配色及页面文字联接已适配 | [Prompt Spider UI](https://github.com/Kirito14IT/prompt-spider-ui)；MIT，原始许可见 [LICENSE](frontend/assets/licenses/prompt-spider-ui.LICENSE.txt) |

## 文献与展示图片

论文元数据保留原始标题、作者、ACL Anthology 页面及 PDF URL，见 [论文清单](data/paper_manifest.json)。论文全文、批量公式裁剪和 QASPER 原始数据不随代码包发布。下载与后续使用遵循各文献来源页面的许可。

README 的界面截图包含 RAGAs 原文页面与公式片段：

> Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. 2024. *RAGAs: Automated Evaluation of Retrieval Augmented Generation*. EACL System Demonstrations. [ACL Anthology](https://aclanthology.org/2024.eacl-demo.16/) · [PDF](https://aclanthology.org/2024.eacl-demo.16.pdf).

该文献的原始页面已在本项目清单中记录为适用 CC BY 4.0。截图仅展示原论文页面或裁剪片段，界面中的缩放不改变原始公式内容。

评测标注中的引文和坐标用于指明原始证据来源，不构成对论文全文或全部公式图像的重新发布。根目录许可证不替代第三方依赖、模型和文献的适用许可。
