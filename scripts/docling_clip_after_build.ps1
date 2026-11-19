param([Parameter(Mandatory)][int]$PipelinePid,
      [Parameter(Mandatory)][ValidatePattern('^[0-9a-f]{16}$')][string]$ParseProfile)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = '1'
$pipeline = Get-Process -Id $PipelinePid -ErrorAction SilentlyContinue
if ($pipeline) { $pipeline.WaitForExit() }
$reportPath = Join-Path $root "runtime/docling-acl/$ParseProfile/OBJECTS_BUILD_INPUT.json"
if (-not (Test-Path -LiteralPath $reportPath)) { throw 'Full object preparation did not publish a report.' }
$objects = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
$indexPath = Join-Path $root "runtime/indexes/paperrag_docling_acl_$($objects.profile_id)/CURRENT.json"
if (-not (Test-Path -LiteralPath $indexPath)) { throw 'The full text index failed or has not been published.' }
$index = Get-Content -LiteralPath $indexPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($index.object_snapshot_id -ne $objects.snapshot_id) { throw 'Full index and object snapshot differ.' }
$python = Join-Path $root '.venv-rag/Scripts/python.exe'
& $python -u scripts/build_docling_clip.py --profile $objects.profile_id --device cuda
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'CUDA CLIP encoding failed; retrying on CPU.'
    & $python -u scripts/build_docling_clip.py --profile $objects.profile_id --device cpu
}
exit $LASTEXITCODE
