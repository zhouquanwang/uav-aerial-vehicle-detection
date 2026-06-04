# Reinstall CUDA PyTorch 2.6.0+cu124 into project .venv (Windows).
# Run when download.pytorch.org is reachable and nvidia-smi shows a GPU.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

& ".\.venv\Scripts\python.exe" -m pip uninstall -y torch torchvision
& ".\.venv\Scripts\python.exe" -m pip install torch==2.6.0 torchvision==0.21.0 `
    --index-url https://download.pytorch.org/whl/cu124
& ".\.venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
