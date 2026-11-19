param([ValidatePattern('^[0-9a-f]{16}$')][string]$Profile,[int]$Threads=8,[int]$WaitForPreparationPid=0,
      [switch]$Objects,[switch]$CudaResume,[switch]$VisualIndex)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = '1'
$parsePython = Join-Path $root 'runtime/docling-venv/Scripts/python.exe'
$indexPython = Join-Path $root '.venv-rag/Scripts/python.exe'
if ($CudaResume) {
    if (-not $Profile -or -not $Objects -or $WaitForPreparationPid) { throw 'CUDA resume needs an object preparation profile and no other parser' }
    $cudaPython = Join-Path $root 'runtime/docling-cuda-venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $cudaPython) {
        & $cudaPython -W 'ignore::FutureWarning' -u scripts/docling_prepare.py --device cuda --threads $Threads --resume-profile $Profile --fail-fast
        $cudaExit = $LASTEXITCODE
    } else { $cudaExit = 1 }
    if ($cudaExit -ne 0) {
        Write-Warning "CUDA parsing stopped (exit $cudaExit); resuming the same cache on CPU."
        & $parsePython -W 'ignore::FutureWarning' -u scripts/docling_prepare.py --device cpu --threads $Threads --resume-profile $Profile
        if ($LASTEXITCODE -ne 0) { Write-Warning 'Check CPU parse results. Object validation will reject missing PDF parses.' }
    }
}
if ($WaitForPreparationPid) {
    if (-not $Profile) { throw 'A preparation profile is required when waiting for a running batch' }
    $preparation = Get-Process -Id $WaitForPreparationPid -ErrorAction SilentlyContinue
    if ($preparation) { $preparation.WaitForExit() }
    if (-not $Objects) {
        $reportPath = Join-Path $root "runtime/docling-acl/$Profile/prepare-full.json"
        if (-not (Test-Path -LiteralPath $reportPath)) { throw 'The preparation batch did not write a complete report' }
        $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($report.failed.Count -gt 0 -or $report.completed -ne $report.requested_papers.Count) {
            throw "Docling preparation did not finish for all papers. See $reportPath"
        }
    }
}
if (-not $Profile) {
    & $parsePython -u scripts/docling_prepare.py --threads $Threads
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $report = Get-ChildItem -LiteralPath (Join-Path $root 'runtime/docling-acl') -Filter 'prepare-full.json' -File -Recurse |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $report) { throw 'No complete Docling preparation report' }
    $Profile = ($report | Get-Content -Raw -Encoding UTF8 | ConvertFrom-Json).profile_id
}
if ($Objects) {
    $objectReport = Join-Path $root "runtime/docling-acl/$Profile/OBJECTS_BUILD_INPUT.json"
    & $parsePython -u scripts/docling_objects.py --parse-profile $Profile --profile-out $objectReport
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $Profile = (Get-Content -LiteralPath $objectReport -Raw -Encoding UTF8 | ConvertFrom-Json).profile_id
    & $indexPython -u scripts/validate_docling_objects.py --profile $Profile --require-full
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $indexPython -u scripts/smoke_docling_objects.py --profile $Profile
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $indexPython -u scripts/docling_index.py build --profile $Profile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($VisualIndex) {
    & $indexPython -u scripts/build_docling_clip.py --profile $Profile --device cuda
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'CUDA CLIP encoding failed; retrying on CPU.'
        & $indexPython -u scripts/build_docling_clip.py --profile $Profile --device cpu
    }
}
exit $LASTEXITCODE
