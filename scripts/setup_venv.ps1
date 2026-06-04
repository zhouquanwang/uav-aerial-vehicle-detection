# Create .venv and install GPU stack (Windows, project root).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    throw "python not found on PATH"
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt -i $Mirror

Write-Host "Installing PyTorch 2.6.0 (CUDA cu124 if download.pytorch.org is reachable)..."
$torchOk = $false
try {
    & ".\.venv\Scripts\python.exe" -m pip install torch==2.6.0 torchvision==0.21.0 `
        --index-url https://download.pytorch.org/whl/cu124 2>$null
    if ($LASTEXITCODE -eq 0) { $torchOk = $true }
} catch {}

if (-not $torchOk) {
    Write-Host "cu124 install failed; falling back to CPU torch from mirror."
    & ".\.venv\Scripts\python.exe" -m pip install torch==2.6.0 torchvision==0.21.0 -i $Mirror
}

& ".\.venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('cuda:', torch.cuda.is_available())"
Write-Host "Done. Activate: .\.venv\Scripts\Activate.ps1"
Write-Host "Smoke test: python scripts\smoke_test.py"
