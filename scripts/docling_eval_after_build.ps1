param([Parameter(Mandatory)][int]$PipelinePid,
      [ValidatePattern('^[0-9a-f]{16}$')][string]$Profile='8bde7f9ad04def90')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = '1'
$pipeline = Get-Process -Id $PipelinePid -ErrorAction SilentlyContinue
if ($pipeline) { $pipeline.WaitForExit() }
$python = Join-Path $root '.venv-rag/Scripts/python.exe'
& $python -u scripts/evaluate_docling.py --profile $Profile --run
exit $LASTEXITCODE
