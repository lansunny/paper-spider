"""Paper Spider：启动本地研究工作台或运行检索与索引命令。"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['web', 'search', 'baseline', 'parse', 'models', 'index', 'ask', 'evaluate', 'doctor'], default='web')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--query', default='RAGAs faithfulness context relevance')
    args, remaining = parser.parse_known_args()
    if args.task in {'models', 'index', 'ask', 'evaluate', 'doctor'}:
        from rag.cli import main as rag_main
        return rag_main([args.task, '--query', args.query, *remaining])
    if remaining:
        parser.error('unrecognized arguments: ' + ' '.join(remaining))
    if args.task == 'web':
        import uvicorn
        print(f'Paper Spider: http://127.0.0.1:{args.port}', flush=True)
        uvicorn.run('backend.app:app', host='127.0.0.1', port=args.port)
        return 0
    script = {'search': 'search_papers.py', 'baseline': 'qasper_baseline.py', 'parse': 'parse_papers.py'}[args.task]
    command = [sys.executable, str(ROOT / 'scripts' / script)]
    if args.task == 'search':
        command.append(args.query)
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
