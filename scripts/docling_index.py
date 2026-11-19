"""Build or query the separate Docling ACL collection with the existing BGE models."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.docling_index import build, search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    ingest = commands.add_parser('build')
    ingest.add_argument('--paper-id', nargs='+', help='Build a separate pilot collection for this subset')
    query = commands.add_parser('search')
    query.add_argument('question')
    query.add_argument('--pilot', action='store_true')
    query.add_argument('--top-k', type=int, default=5)
    query.add_argument('--no-rerank', action='store_true')
    query.add_argument('--paper-id', help='Scope search and enable exact figure/table number lookup')
    for command in (ingest, query):
        command.add_argument('--profile', required=True)
        command.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    if args.command == 'build':
        result = build(args.profile, paper_ids=args.paper_id, device=args.device)
        result = {**result, 'documents': len(result['documents'])}
    else:
        if not args.question.strip() or not 1 <= args.top_k <= 20:
            parser.error('search needs a nonempty question and --top-k in 1..20')
        result = search(args.profile, args.question, pilot=args.pilot, device=args.device,
                        top_k=args.top_k, rerank=not args.no_rerank, paper_id=args.paper_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
