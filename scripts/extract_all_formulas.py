"""PyCharm batch entry: no parameters means all downloaded/uploaded papers.

Small sample: --limit 5
Preview only: --dry-run
Resume: run the same command again; cached results are validated and reused.
Retry failures: --retry-failed
This script performs native-PDF extraction only, without model or API calls.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import portalocker
from rag.formula_batch import run_formula_batch, select_papers, catalog_counts, batch_state_path
from rag.ingestion import paper_catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-id', nargs='+', help='Optional explicit subset; otherwise the complete catalog')
    parser.add_argument('--limit', type=int, default=0, help='0 = all papers; positive values select a small sample')
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Preview catalog selection without extracting or writing files')
    parser.add_argument('--state-file', type=Path, help='Optional isolated checkpoint under runtime/formula_batch')
    args = parser.parse_args(argv)
    try:
        selected = select_papers(paper_catalog(ROOT), args.paper_id, args.limit)
        state_path = batch_state_path(ROOT, args.state_file)
        if not 72 <= args.dpi <= 400:
            parser.error('--dpi must be between 72 and 400')
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(f'目录选中 {len(selected)} 篇：{catalog_counts(selected)}')
        print('前 5 篇：', ', '.join(p['paper_id'] for p in selected[:5]))
        print('抽取全部页；成功后发布到：', ROOT/'runtime/formulas/CURRENT.json')
        print('进度文件：', state_path)
        if args.retry_failed:
            print('--retry-failed 会在运行时按上述进度文件进一步筛选失败论文。')
        print('预览结束；未抽取、未写文件、未调用模型。')
        return 0
    if not selected:
        print('目录没有可抽取的论文。')
        return 1
    try:
        result = run_formula_batch(selected, root=ROOT, dpi=args.dpi, retry_failed=args.retry_failed,
                                   state_path=state_path, emit=lambda message: print(message, flush=True))
    except portalocker.exceptions.LockException:
        print('已有任务占用此进度文件；请等待该任务完成。', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('任务已停止；已完成的论文保留，重新运行同一入口即可续跑。')
        return 130
    except (ValueError, OSError) as error:
        print('批量任务无法启动或保存进度：'+str(error), file=sys.stderr)
        return 1
    print(f"汇总：完成 {result['completed']}/{result['total']} 篇（缓存 {result['cached']}），"
          f"失败 {result['failed']}，候选 {result['formula_candidates']}，编号 {result['numbered_candidates']}。")
    print('需要外部版面/OCR 的论文数：', len(result['papers_needing_layout_ocr']))
    print('进度与失败详情：', result['state_path'])
    print('结果均为未经人工核验的规则候选；本任务不执行 OCR、不建立公式向量索引。')
    return 130 if result['status'] == 'interrupted' else 1 if result['failed'] else 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
