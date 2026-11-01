"""Join formulas to an existing text retrieval trace, without models or API calls.

PyCharm: --paper-id 2024.eacl-demo.16 --question "公式 (1)"
Actual chunks: --trace runtime/traces/<trace>.json --question "your question"
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.formula_retrieval import attach_formulas
from rag.storage import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', required=True)
    parser.add_argument('--paper-id', nargs='+')
    parser.add_argument('--trace', type=Path)
    args=parser.parse_args()
    if not args.paper_id and not args.trace:
        parser.error('Choose --paper-id or supply a text retrieval --trace')
    trace={}
    if args.trace:
        path=args.trace if args.trace.is_absolute() else ROOT/args.trace
        trace=json.loads(path.read_text(encoding='utf-8'))
    result=attach_formulas(args.question,trace.get('evidence',[]),
                           paper_ids=args.paper_id or trace.get('route',{}).get('paper_ids'),root=ROOT)
    path=ROOT/'runtime/formula_learning/latest.json'
    atomic_json(path,{'question':args.question,**result})
    for f in result['formulas']:
        print(f['paper_id'],f['pdf_page'],f['equation_number'] or '-',f['element_id'][:8],
              ','.join(m['method'] for m in f['matches']))
    print('Formula vector index: False; OCR: False; images are original crops.')
    print('Result:',path)
    print('Diagnostics:',result['formula_retrieval']['diagnostics'])


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
