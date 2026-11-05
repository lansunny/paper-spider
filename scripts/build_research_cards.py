"""Build cached source-linked research cards for the current graph's topic communities."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.reviews.cards import build_cards


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    manifest = build_cards(ROOT, progress=lambda **s: print(s, flush=True))
    print('Completed:', manifest['topic_cards'], 'topic cards;', manifest['method_routes'], 'method routes', flush=True)
