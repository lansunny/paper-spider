"""Run the fixed original-PDF formula regression labels; no model calls."""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rag.formula_evaluation import evaluate_formula_regression
from rag.storage import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,default=ROOT/'data/evaluation/formula_regression_v1.json')
    args=parser.parse_args()
    path=args.annotations if args.annotations.is_absolute() else ROOT/args.annotations
    result=evaluate_formula_regression(path,ROOT)
    target=ROOT/'runtime/formula_evaluation/latest.json'; atomic_json(target,result)
    print(f"Known positives retained: {result['positive_retained']}/{result['positive_cases']}")
    print(f"Known negatives rejected: {result['negative_rejected']}/{result['negative_cases']}")
    print('Failures:',[r['annotation_id'] for r in result['results'] if not r['passed']])
    print('Hard-case regression only; corpus-wide precision/recall is unknown.')
    print('Result:',target)
    return 0 if result['all_passed'] else 1


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
