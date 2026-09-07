param(
    [string]$Python = "python",
    [switch]$InstallCpuTorch,
    [switch]$InstallCudaTorch,
    [switch]$InstallDev,
    [switch]$InstallApp
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv $venvPath
}

& $venvPython -m pip install --upgrade pip

$requirements = @()
if ($InstallDev) {
    $requirements += Join-Path $projectRoot "requirements-dev.txt"
}
if ($InstallApp) {
    $requirements += Join-Path $projectRoot "requirements-app.txt"
}
if ($requirements.Count -eq 0) {
    $requirements += Join-Path $projectRoot "requirements-base.txt"
}

foreach ($requirementsFile in $requirements) {
    & $venvPython -m pip install -r $requirementsFile
}

if ($InstallCpuTorch -and $InstallCudaTorch) {
    throw "-InstallCpuTorch와 -InstallCudaTorch를 동시에 사용할 수 없습니다."
}

if ($InstallCudaTorch) {
    & $venvPython -m pip install -r (Join-Path $projectRoot "requirements-torch-cu130.txt")
} elseif ($InstallCpuTorch) {
    & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    Write-Host "PyTorch는 아직 설치하지 않았습니다."
    Write-Host "GPU 사용자는 https://pytorch.org/get-started/locally/ 의 선택 명령을 실행하세요."
}

& $venvPython -m pip install --no-deps --no-build-isolation -e $projectRoot
& $venvPython (Join-Path $projectRoot "scripts\doctor.py")
