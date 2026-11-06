"""PyCharm: run with no parameters to test all 50 curated formula questions.

--validate-only checks labels/artifacts without loading models or Qdrant.
--limit 3 performs a small smoke run; repeating the same profile reuses results.
--config configs/retrieval_structure.json evaluates the structural text index.
Default V2 uses the Chinese question, glossary expansion and per-paper facets.
--profile v1 preserves the old English-assisted, 8000-character baseline.
No interactive input, answer-generation API, OCR, or formula vector index.
"""
import argparse
import sys
from urllib.parse import urlsplit
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from rag.config import load_settings
from rag.formula_benchmark import load_dataset
from rag.formula_experiment import preflight,run_experiment
from rag.formula_store import FormulaStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--profile',choices=['v1','v2'],default='v2')
    parser.add_argument('--no-decomposition',action='store_true')
    parser.add_argument('--no-lexical-expansion',action='store_true')
    parser.add_argument('--no-formula-rank',action='store_true')
    parser.add_argument('--no-formula-rescue',action='store_true',help='Ablate source-aware preservation before the 20-candidate trim')
    parser.add_argument('--context-budget',type=int,help='Serialized evidence character limit; not model tokens')
    parser.add_argument('--limit',type=int,default=0,help='0 = all 50 questions')
    parser.add_argument('--question-id',nargs='+')
    parser.add_argument('--mode',choices=['hybrid','bm25','dense'],default='hybrid')
    parser.add_argument('--no-rerank',action='store_true')
    lexical=parser.add_mutually_exclusive_group()
    lexical.add_argument('--original-lexical-query',action='store_true',help='Use original Chinese question (V2 default)')
    lexical.add_argument('--english-lexical-query',action='store_true',help='Explicit English-assisted diagnostic; not real Chinese deployment')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--show-formulas',type=int,default=2,help='Number of returned image paths to print per question')
    args=parser.parse_args()
    if args.limit<0 or args.show_formulas<0:parser.error('limits must be nonnegative')
    if args.context_budget is not None and args.context_budget<1000:parser.error('context budget must be >=1000')
    if args.profile=='v1' and any([args.no_decomposition,args.no_lexical_expansion,args.no_formula_rank,args.no_formula_rescue]):
        parser.error('V2 ablations require --profile v2')
    config=args.config if not args.config or args.config.is_absolute() else ROOT/args.config
    if args.profile=='v2':
        from rag.pipeline_v2 import load_v2_settings,RetrievalPolicyV2
        settings=load_v2_settings(config)
        policy=RetrievalPolicyV2(decomposition=not args.no_decomposition,
                                 lexical_expansion=not args.no_lexical_expansion,
                                 formula_ranking=not args.no_formula_rank and not args.no_rerank,
                                 formula_rescue=not args.no_formula_rescue)
    else:
        settings=load_settings(config);policy=None
    if args.context_budget is not None:
        settings=settings.model_copy(update={'context_char_budget':args.context_budget})
    english_lexical=args.english_lexical_query or (args.profile=='v1' and not args.original_lexical_query)
    questions,gold,meta=load_dataset(ROOT/'data/evaluation/formula_test50',ROOT)
    if len(questions)!=50:raise ValueError('The prepared formula test50 must contain exactly 50 questions')
    if args.question_id:
        wanted=set(args.question_id)
        if wanted-{q.question_id for q in questions}:parser.error('unknown question ID')
        questions=[q for q in questions if q.question_id in wanted]
    elif args.limit:questions=questions[:args.limit]
    store=FormulaStore(ROOT);manifest,available=preflight(settings,gold,ROOT,store)
    print(f"50题数据已校验：{meta['gold_formulas']} 个原文公式位置，{meta['source_papers']} 篇论文；当前公式库 {len(store.records)} 篇。",flush=True)
    print('题型：',meta['categories'])
    print(f"原文标准公式当前可定位：{sum(a['available'] for a in available.values())}/{len(available)}；文本索引 {manifest['snapshot_id']}。")
    print(f'检索方案：{args.profile}；上下文预算：{settings.context_char_budget}字符；全部重排候选参与筛选。')
    print('标准答案仅供评分，不传入检索；BM25 使用'+('已标注的英文等义问题（辅助诊断）。' if english_lexical else '原始中文问题'+('＋科学术语扩展。' if args.profile=='v2' and policy.lexical_expansion else '。')))
    if args.validate_only:
        print('校验完成；未加载模型、未执行检索。');return
    if args.mode in {'hybrid','dense'}:
        # Clear prerequisite error before loading local models or starting a run.
        import socket
        endpoint=urlsplit(settings.qdrant_url)
        if endpoint.hostname in {'127.0.0.1','localhost','::1'}:
            port=endpoint.port or (443 if endpoint.scheme=='https' else 80)
            try:
                with socket.create_connection((endpoint.hostname,port),timeout=2):pass
            except OSError:
                parser.exit(1,'本地 Qdrant 未启动。请先启动已有的 Paper Spider Qdrant 服务，再运行混合检索测试。\n'
                              '也可用 --mode bm25 --no-rerank 验证离线 BM25 路径。\n')

    def show(number,total,q,row,action):
        print(f'[{number}/{total}] {q.question_id} {q.category} ({action}) {q.question}',flush=True)
        if q.expectation=='answerable':
            stages=['union','rrf20','after_top5','context_budget']
            values=' -> '.join(f"{row['metrics'][s]['anchor_recall']:.1%}" for s in stages)
            top5='Top5（未重排）' if args.no_rerank else '重排Top5'
            print(f'  公式锚点召回（候选全集→候选截取→{top5}→预算上下文）：'+values+
                  f"；全部目标齐全={row['metrics']['context_budget']['all_required']}；路由={row['route']['mode']}",flush=True)
        else:
            print(f"  控制项通过={row['metrics']['context_budget']['control_passed']}；附带公式={len(row['final_formulas'])}；不代表模型拒答正确。",flush=True)
        print(f"  上下文 {row['context_items']}/{row['context_candidate_items']} 条，{row['context_chars']}/{row['context_budget']} 字符；"
              f"子问题 {len((row.get('query_plan') or {}).get('facets',[]))} 个。",flush=True)
        for f in row['final_formulas'][:args.show_formulas]:
            methods=','.join(sorted({m['method'] for m in f['matches']}))
            print(f"  {f['title']} | PDF {f['pdf_page']} | Eq {f['equation_number'] or '-'} | {methods}")
            print('  原图：'+f['image_path'])
        if row['failure_diagnosis']:print('  未命中原因：',row['failure_diagnosis'])
    target,summary=run_experiment(questions,gold,meta,settings,ROOT,mode=args.mode,rerank=not args.no_rerank,
                                  english_lexical=english_lexical,emit=show,retrieval_profile=args.profile,policy=policy)
    for cohort in ['semantic_main','lookup_diagnostics']:
        data=summary[cohort];metrics=data['stages'].get('context_budget')
        if metrics:print(f"{cohort} | {data['questions']}题 | 预算后公式锚点召回 {metrics['macro_formula_anchor_recall']:.1%}"
                         f" | 全部目标齐全比例 {metrics['all_required_question_rate']:.1%}"
                         f" | 仅正文联接召回 {metrics['join_only_anchor_recall']:.1%}")
    controls=summary['by_category'].get('control')
    if controls:print(f"控制项：{controls['controls_passed']}/{controls['controls']} 通过。")
    print('结果：',target/'summary.json')
    print('逐题 JSON 包含标准答案、来源锚点匹配、失败阶段与返回原图路径。只评估公式召回，不评估生成回答。')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
