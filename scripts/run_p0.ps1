param(
    [switch]$AggregateOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$splits = Join-Path $projectRoot "artifacts\splits.csv"
$expectedSplitHash = "11fa5f986749f905e6b498b0a40bd4464c0575d6f73fa41215f6269b79f36090"
$experiments = @("E1", "E2", "E3")
$seeds = @(42, 52, 62, 72, 82)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual-environment Python not found: $python"
}
if (-not (Test-Path -LiteralPath $splits -PathType Leaf)) {
    throw "Split file not found: $splits"
}

$actualSplitHash = (Get-FileHash -LiteralPath $splits -Algorithm SHA256).Hash.ToLower()
if ($actualSplitHash -ne $expectedSplitHash) {
    throw "Split SHA-256 mismatch. Expected $expectedSplitHash, got $actualSplitHash."
}

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

        foreach ($experiment in $experiments) {
            foreach ($seed in $seeds) {
                $metrics = Join-Path $projectRoot (
                    "artifacts\repeat\$experiment\seed$seed\validation_metrics.json"
                )
                if (Test-Path -LiteralPath $metrics -PathType Leaf) {
                    Write-Host "SKIP completed: $experiment seed=$seed"
                    continue
                }
                Write-Host "RUN: $experiment seed=$seed"
                & $python ".\scripts\run_experiment.py" `
                    "--experiment" $experiment `
                    "--seed" $seed
                if ($LASTEXITCODE -ne 0) {
                    throw "P0 run failed: $experiment seed=$seed"
                }
            }
        }
    }

    & $python ".\scripts\aggregate_seeds.py"
    if ($LASTEXITCODE -ne 0) {
        throw "P0 aggregation failed."
    }
}
finally {
    Pop-Location
}
