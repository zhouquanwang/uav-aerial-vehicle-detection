"""Entry launcher: runs inference with the project .venv when available."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _python() -> Path:
    if sys.platform == "win32":
        venv = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv = ROOT / ".venv" / "bin" / "python"
    return venv if venv.is_file() else Path(sys.executable)


def main() -> None:
    py = _python()
    cmd = [str(py), "-m", "src.predict_video", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
