"""Encode original Docling figure crops for the selected released index."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.docling_index import settings_for
from rag.figure_vectors import CLIP_MODEL, CLIP_PATH, CLIP_REVISION, LocalClip, build_visual
from rag.ingestion import current_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile')
    parser.add_argument('--download-model', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    if args.download_model:
        from huggingface_hub import snapshot_download
        snapshot_download(CLIP_MODEL, revision=CLIP_REVISION, local_dir=CLIP_PATH, allow_patterns=[
            'model.safetensors', 'config.json', 'preprocessor_config.json', 'tokenizer.json',
            'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json'])
        if not args.profile:
            return
    if not args.profile:
        parser.error('Specify --profile to build the visual index')
    _, cfg = settings_for(args.profile, pilot=args.pilot)
    manifest = current_manifest(cfg)
    if not manifest or not manifest.get('object_db') or manifest['profile'] != cfg.index_profile():
        parser.error('Publish the Docling object index before building its visual index')
    report = build_visual(args.profile, manifest, pilot=args.pilot, encoder=LocalClip(args.device))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
