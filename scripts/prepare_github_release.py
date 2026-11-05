"""Create a local, allowlisted source snapshot and ZIP without publishing anything."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ('README.md','LICENSE','NOTICE','THIRD_PARTY_NOTICES.md','.env.example',
              '.gitignore','.gitattributes','main.py','compose.yaml','framework.json',
              'requirements.txt','requirements-base.txt','requirements-graph.txt')
PATTERNS = ('rag/**/*.py','backend/**/*.py','frontend/**/*.html','frontend/**/*.css',
            'frontend/**/*.js','frontend/**/*.svg','frontend/**/*.ttf','frontend/**/OFL.txt','frontend/assets/licenses/*.txt',
            'frontend/assets/pdfjs/**/*.mjs','frontend/assets/pdfjs/**/*.bcmap','frontend/assets/pdfjs/**/*.pfb',
            'frontend/assets/pdfjs/**/*.otf','frontend/assets/pdfjs/**/*.wasm','frontend/assets/pdfjs/LICENSE',
            'frontend/assets/pdfjs/**/LICENSE*','frontend/assets/pdfjs/version.json',
            'scripts/**/*.py','tests/**/*.py','configs/*.json','.github/workflows/*.yml',
            'docs/assets/**/*.svg','docs/assets/**/*.jpg','docs/assets/**/*.gif','docs/assets/**/*.png','docs/benchmarks/*.json',
            'data/evaluation/paper_benchmark/*.jsonl')
DOCS = ('getting-started.md','architecture.md','evaluation.md','data.md',
        'community-evidence-experiment.md','community-evidence-results.md')
DATA = ('data/paper_catalog.json','data/paper_manifest.json',
        'data/evaluation/paper_benchmark/manifest.json',
        'data/evaluation/paper_benchmark/test50_manifest.json',
        'data/evaluation/formula_test50/curation.json',
        'data/evaluation/formula_test50/questions.jsonl')
TOKEN = re.compile(r'(?<![A-Za-z0-9])(?:sk-|github_pat_|ghp_|gho_)[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
MAX_FILE_BYTES = 5 * 1024 * 1024


def public_files(root):
    root = root.resolve()
    files = {root/p for p in (*ROOT_FILES,*DATA,*(str(Path('docs')/p) for p in DOCS)) if (root/p).is_file()}
    for pattern in PATTERNS:
        files.update(p for p in root.glob(pattern) if p.is_file())
    for path in files:
        rel = path.relative_to(root)
        if path.is_symlink() or not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in path.parents if p!=root):
            raise ValueError('Linked files are not allowed in a release: '+rel.as_posix())
        if any(part.startswith('.env') and part!='.env.example' for part in rel.parts):
            raise ValueError('Environment file is not publishable: '+rel.as_posix())
        if any(part in {'__pycache__','.git','runtime','vendor','resume','tmp'} for part in rel.parts):
            raise ValueError('Private or generated directory selected: '+rel.as_posix())
        if path.stat().st_size>MAX_FILE_BYTES:
            raise ValueError('Unexpectedly large public file: '+rel.as_posix())
        if path.suffix.lower() not in {'.jpg','.jpeg','.png','.gif','.ttf','.bcmap','.pfb','.otf','.wasm'}:
            text=path.read_text(encoding='utf-8')
            if TOKEN.search(text):
                raise ValueError('Credential-like content found; export stopped: '+rel.as_posix())
            if path.name=='.env.example':
                for line in text.splitlines():
                    if line.startswith('DEEPSEEK_API_KEY=') and line.split('=',1)[1].strip():
                        raise ValueError('Example API key must be blank')
    return sorted(files,key=lambda p:p.relative_to(root).as_posix())


def prepare_release(root, target):
    root, target = root.resolve(),target.resolve()
    parent=(root/'runtime/github-preview').resolve()
    if target==parent or not target.is_relative_to(parent):
        raise ValueError('Release output must be a new directory under runtime/github-preview')
    archive=Path(str(target)+'.zip')
    if target.exists() or archive.exists():
        raise ValueError('Output already exists; choose a new output name instead of overwriting')
    files=public_files(root)
    if not (root/'README.md').is_file() or not (root/'main.py').is_file():
        raise ValueError('Expected a Paper spider source directory')
    target.mkdir(parents=True)
    records=[]
    for source in files:
        relative=source.relative_to(root)
        output=target/relative;output.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,output)
        records.append({'path':relative.as_posix(),'bytes':output.stat().st_size,
                        'sha256':hashlib.sha256(output.read_bytes()).hexdigest()})
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in records:
            z.write(target/row['path'],'paper-spider/'+row['path'])
    report={'status':'source_snapshot','files':len(records),'uncompressed_bytes':sum(r['bytes'] for r in records),
            'archive_bytes':archive.stat().st_size,'contents':records,
            'excluded':['credentials and local environment','PDF corpus and formula crops','model weights and indexes',
                        'runtime traces and uploaded files','personal notes and documents','reference vendor source'],
            'github_actions_performed':False}
    manifest=Path(str(target)+'.manifest.json')
    manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return target,archive,manifest,report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runtime/github-preview/paper-spider')
    parser.add_argument('--list-only',action='store_true')
    args=parser.parse_args(argv)
    if args.list_only:
        files=public_files(ROOT)
        print(json.dumps({'files':len(files),'bytes':sum(p.stat().st_size for p in files),
                          'paths':[p.relative_to(ROOT).as_posix() for p in files]},ensure_ascii=False,indent=2))
        return 0
    target=args.output if args.output.is_absolute() else ROOT/args.output
    directory,archive,manifest,report=prepare_release(ROOT,target)
    print(json.dumps({'directory':str(directory),'archive':str(archive),'manifest':str(manifest),
                      'files':report['files'],'archive_bytes':report['archive_bytes'],
                      'uploaded':False},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
