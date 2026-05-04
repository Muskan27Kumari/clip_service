"""
Typo helper: run ``video_streamer`` with the project venv when present.
"""
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_venv_py = _root / "venv" / "bin" / "python"
_target = _root / "video_streamer.py"

if __name__ == "__main__":
    if _venv_py.is_file() and str(os.environ.get("CLIP_SERVICE_USE_VENV", "1")).lower() not in (
        "0",
        "false",
        "no",
    ):
        os.execv(str(_venv_py), [str(_venv_py), str(_target), *sys.argv[1:]])
    os.execv(sys.executable, [sys.executable, str(_target), *sys.argv[1:]])
