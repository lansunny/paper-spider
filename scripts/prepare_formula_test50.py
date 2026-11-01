"""Build the explicitly curated formula test50 labels from original PDF coordinates."""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rag.formula_benchmark import prepare_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force',action='store_true',help='Rebuild from the same fixed source curation')
    args=parser.parse_args()
    meta=prepare_dataset(ROOT/'data/evaluation/formula_test50',ROOT,force=args.force)
    print(f"Prepared {meta['questions']} questions, {meta['gold_formulas']} original formula anchors, {meta['source_papers']} papers.")
    print('Categories:',meta['categories'])
    print('Labels are agent-reviewed; no formula detector or element IDs define the gold locations.')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
