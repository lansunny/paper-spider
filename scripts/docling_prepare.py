"""Parse the active ACL corpus with Docling in its own Python environment."""
import argparse
from collections import Counter
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.storage import atomic_json, atomic_text, read_json

BASE = ROOT / 'runtime/indexes/paperrag_bge_m3_v1/CURRENT.json'
WORK = ROOT / 'runtime/docling-acl'
TOKENS = 384
MAX_EMBEDDING_TOKENS = 512
BGE = ROOT / 'runtime/models/embedding'


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def stable_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def sources():
    manifest = read_json(BASE)
    files = {row['paper_id']: row for row in read_json(ROOT / 'data/paper_manifest.json')
             if row['status'] == 'downloaded'}
    ids = manifest['documents']
    if set(ids) != set(files):
        raise ValueError('ACL manifest and current baseline index contain different paper IDs')
    return manifest, files


def artifact(profile_id, paper_id, pdf_hash):
    # Neither the user-supplied ID nor filename is used to form the directory name.
    return WORK / profile_id / stable_hash([paper_id, pdf_hash])[:32]


def prepare(args):
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this Docling environment')
        print(f'Docling CUDA: {torch.cuda.get_device_name(0)} ({torch.__version__})', flush=True)
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer
    from docling_core.types.doc import DoclingDocument

    manifest, documents = sources()
    ids = args.paper_id or sorted(documents)
    if len(ids) != len(set(ids)) or not set(ids).issubset(documents):
        raise ValueError('Expected unique paper IDs from the active ACL baseline')
    if args.limit:
        ids = ids[:args.limit]
    options = PdfPipelineOptions(do_ocr=args.ocr, do_table_structure=True,
                                 heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
                                 accelerator_options=AcceleratorOptions(device=args.device, num_threads=args.threads))
    tokenizer = AutoTokenizer.from_pretrained(BGE, local_files_only=True)
    chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=args.max_tokens),
                           merge_peers=True, repeat_table_header=True)
    abstract_options = options.model_dump(mode='json', exclude={'accelerator_options', 'artifacts_path'})
    parse_profile = {'pdf_options': abstract_options, 'software': {p: version(p) for p in (
        'docling-slim', 'docling-core', 'docling-parse', 'docling-ibm-models', 'transformers', 'torch')}}
    if args.resume_profile:
        previous = read_json(WORK / args.resume_profile / 'profile.json')
        if not previous or stable_hash(previous)[:16] != args.resume_profile or previous['format_version'] != 2:
            raise ValueError('Cannot resume an unknown Docling preparation profile')
        prior_parser = previous['parser']

        def comparable(parser):
            software = parser['software']
            pdf_options = dict(parser['pdf_options'])
            # Docling serializes this engine set in hash-dependent order.
            if model := pdf_options.get('chart_extraction_options', {}).get('model_spec'):
                pdf_options['chart_extraction_options'] = {**pdf_options['chart_extraction_options'], 'model_spec': {
                    **model, 'supported_engines': sorted(model['supported_engines'])}}
            return {**parser, 'pdf_options': pdf_options,
                    'software': {**software, 'torch': software['torch'].split('+', 1)[0]}}

        if comparable(prior_parser) != comparable(parse_profile):
            raise ValueError('Resume requires the same PDF settings and package versions except the PyTorch accelerator build')
        parse_profile = prior_parser
    parse_id = stable_hash(parse_profile)[:16]
    profile = {'format_version': 2, 'parse_id': parse_id, 'parser': parse_profile, 'options_chunker': {
        'algorithm': 'HybridChunker', 'max_tokens': args.max_tokens, 'merge_peers': True, 'repeat_table_header': True},
        'tokenizer_sha256': digest(BGE / 'tokenizer.json'),
        'bge': read_json(BGE / 'paperrag-model.json')}
    profile_id = stable_hash(profile)[:16]
    if args.resume_profile and profile_id != args.resume_profile:
        raise ValueError('Current tokenizer or chunker settings differ from the resume profile')
    atomic_json(WORK / profile_id / 'profile.json', profile)
    print(f'Corpus {manifest["snapshot_id"]}; profile {profile_id}; {len(ids)} papers', flush=True)
    converter = None
    fresh_conversions = 0
    rows, failures = [], []
    for i, pid in enumerate(ids, 1):
        pdf = (ROOT / documents[pid]['pdf_path']).resolve()
        if not pdf.is_relative_to(ROOT.resolve()):
            raise ValueError('PDF escaped the project directory')
        pdf_hash = manifest['documents'][pid]
        cache = artifact(profile_id, pid, pdf_hash)
        parsed = artifact('parsed/' + parse_id, pid, pdf_hash)
        try:
            if digest(pdf) != pdf_hash:
                raise ValueError('PDF version differs from the active ACL baseline')
            info = read_json(cache / 'manifest.json')
            if info and info['profile_id'] == profile_id and info['paper_id'] == pid and info['pdf_sha256'] == pdf_hash:
                if ((parsed / 'document.json').is_file() and digest(parsed / 'document.json') == info['sha256']['document.json']
                        and (cache / 'chunks.jsonl').is_file() and digest(cache / 'chunks.jsonl') == info['sha256']['chunks.jsonl']):
                    rows.append(info)
                    print(f'[{i}/{len(ids)}] {pid}: cached {info["chunks"]}', flush=True)
                    continue
            start = perf_counter()
            parsed_file = parsed / 'document.json'
            parse_meta = read_json(parsed / 'manifest.json')
            if parse_meta and parsed_file.is_file() and digest(parsed_file) == parse_meta['sha256']:
                doc = DoclingDocument.load_from_json(parsed_file)
            else:
                if converter is None:
                    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
                                                  allowed_formats=[InputFormat.PDF])
                conversion = converter.convert(pdf, raises_on_error=False)
                if conversion.status.value != 'success':
                    raise RuntimeError(f'Docling status: {conversion.status.value}; {conversion.errors[:2]}')
                doc = conversion.document
                if digest(pdf) != pdf_hash:
                    raise ValueError('PDF changed during parsing')
                atomic_json(parsed_file, doc.export_to_dict())
                atomic_json(parsed / 'manifest.json', {'sha256': digest(parsed_file), 'device': args.device,
                                                       'torch': version('torch')})
                fresh_conversions += 1
            chunks, counts, notices = [], Counter(), Counter()

            def append(text, embed, headings, items, kind):
                pages = sorted({prov['page_no'] for item in items for prov in item['provenance']})
                if not pages or any(page not in doc.pages for page in pages):
                    notices['without_page_provenance'] += 1
                    return
                length = len(tokenizer.encode(embed, add_special_tokens=True, truncation=False))
                if length > MAX_EMBEDDING_TOKENS:
                    raise ValueError(f'Chunk uses {length} embedding tokens; limit is {MAX_EMBEDDING_TOKENS}')
                chunks.append({'chunk_index': len(chunks), 'text': text, 'embedding_text': embed,
                               'tokens': length, 'headings': headings, 'kind': kind,
                               'elements': items, 'pages': pages})
                counts[kind] += 1

            for chunk in chunker.chunk(dl_doc=doc):
                if not chunk.text.strip():
                    continue
                # These identify the contributing elements, not substring offsets into legacy page text.
                items = [{'id': item.self_ref, 'label': item.label.value,
                          'provenance': [prov.model_dump(mode='json') for prov in item.prov]}
                         for item in chunk.meta.doc_items]
                kind = 'table' if any(x['label'] == 'table' for x in items) else (
                    'figure' if any(x['label'] == 'picture' for x in items) else 'text')
                append(chunk.text, chunker.contextualize(chunk), chunk.meta.headings or [], items, kind)

            # A vector indexes the caption, never a guessed description of the image.
            figure_refs = {item['id'] for chunk in chunks if chunk['kind'] == 'figure' for item in chunk['elements']}
            for picture in doc.pictures:
                if picture.self_ref in figure_refs:
                    continue
                captions = [ref.resolve(doc) for ref in picture.captions]
                text = '\n'.join(item.text for item in captions if getattr(item, 'text', '').strip())
                if not text:
                    notices['figures_without_caption'] += 1
                    continue
                if len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) > MAX_EMBEDDING_TOKENS:
                    notices['figures_with_overlong_caption'] += 1
                    continue
                items = [{'id': item.self_ref, 'label': item.label.value,
                          'provenance': [prov.model_dump(mode='json') for prov in item.prov]}
                         for item in (picture, *captions)]
                append(text, text, [], items, 'figure')
            if not chunks:
                raise ValueError('Docling produced no indexable chunks')
            if digest(pdf) != pdf_hash:
                raise ValueError('PDF changed during parsing')
            atomic_text(cache / 'chunks.jsonl', ''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in chunks))
            info = {'profile_id': profile_id, 'paper_id': pid, 'pdf_sha256': pdf_hash,
                    'pages': len(doc.pages), 'chunks': len(chunks), 'chunk_kinds': dict(counts),
                    'tables': len(doc.tables), 'figures': len(doc.pictures),
                    'notices': dict(notices),
                    'max_tokens': max(c['tokens'] for c in chunks), 'elapsed_seconds': round(perf_counter()-start, 2),
                    'sha256': {'document.json': digest(parsed_file), 'chunks.jsonl': digest(cache / 'chunks.jsonl')}}
            atomic_json(cache / 'manifest.json', info)
            rows.append(info)
            print(f'[{i}/{len(ids)}] {pid}: {len(chunks)} chunks, {len(doc.pages)} pages, {info["elapsed_seconds"]}s', flush=True)
        except Exception as error:
            failures.append({'paper_id': pid, 'type': type(error).__name__, 'message': str(error)[:800]})
            print(f'[{i}/{len(ids)}] {pid}: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
            # A first-document environment/model failure should not be retried 500 times.
            if args.fail_fast or (not rows and len(failures) >= min(3, len(ids))):
                break
    report = {'baseline_snapshot': manifest['snapshot_id'], 'profile_id': profile_id,
              'requested_papers': ids, 'completed': len(rows), 'failed': failures, 'papers': rows,
              'requested_device': args.device, 'fresh_conversions': fresh_conversions}
    name = 'prepare-full' if len(ids) == len(documents) else 'prepare-subset-' + stable_hash(ids)[:8]
    atomic_json(WORK / profile_id / (name + '.json'), report)
    return 0 if len(rows) == len(ids) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-id', nargs='+')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--max-tokens', type=int, default=TOKENS)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--resume-profile', help='Reuse a compatible Docling parse cache across CPU and GPU')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--ocr', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or args.limit < 0 or not 64 <= args.max_tokens <= MAX_EMBEDDING_TOKENS - 2:
        parser.error('Specify positive --threads, nonnegative --limit and --max-tokens in 64..510')
    if args.resume_profile and (len(args.resume_profile) != 16 or any(c not in '0123456789abcdef' for c in args.resume_profile)):
        parser.error('Invalid --resume-profile')
    os.environ.setdefault('HF_HOME', str(ROOT / 'runtime/docling-models'))
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(prepare(args))


if __name__ == '__main__':
    main()
