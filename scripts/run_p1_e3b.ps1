param(
    [switch]$AggregateOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$splits = Join-Path $projectRoot "artifacts\splits.csv"
$expectedSplitHash = "11fa5f986749f905e6b498b0a40bd4464c0575d6f73fa41215f6269b79f36090"
$seeds = @(42, 52, 62, 72, 82)
$logPath = Join-Path $projectRoot "artifacts\p1_e3b_run.log"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual-environment Python not found: $python"
}
$actualSplitHash = (Get-FileHash -LiteralPath $splits -Algorithm SHA256).Hash.ToLower()
if ($actualSplitHash -ne $expectedSplitHash) {
    throw "Split SHA-256 mismatch. Expected $expectedSplitHash, got $actualSplitHash."
}

Start-Transcript -LiteralPath $logPath -Append
Push-Location $projectRoot
try {
    if (-not $AggregateOnly) {
        & $python ".\scripts\doctor.py" "--strict"
        if ($LASTEXITCODE -ne 0) {
            throw "doctor.py --strict failed."
        }
        & $python "-m" "ruff" "check" "."
        if ($LASTEXITCODE -ne 0) {
            throw "ruff check failed."
        }
        & $python "-m" "pytest"
        if ($LASTEXITCODE -ne 0) {
            throw "pytest failed."
        }

        foreach ($seed in $seeds) {
            $metrics = Join-Path $projectRoot (
                "artifacts\repeat\E3b\seed$seed\validation_metrics.json"
            )
            if (Test-Path -LiteralPath $metrics -PathType Leaf) {
                Write-Host "SKIP completed: E3b seed=$seed"
                continue
            }
            Write-Host "RUN: E3b seed=$seed"
            & $python ".\scripts\run_experiment.py" `
                "--experiment" "E3b" `
                "--seed" $seed
            if ($LASTEXITCODE -ne 0) {
                throw "P1 E3b run failed: seed=$seed"
            }
        }
    }

    & $python ".\scripts\aggregate_p1_e3b.py"
    if ($LASTEXITCODE -ne 0) {
        throw "P1 E3b aggregation failed."
    }
}
finally {
    Pop-Location
    Stop-Transcript
}
